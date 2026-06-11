import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch

from visualine.core.config_loader import get_resource_path
from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.grounding_dino_wrapper import GroundingDinoWrapper
from visualine.nodes.utils.box_utils import (
    convert_box_to_xyxy,
    filter_boxes_xyxy,
)

logger = logging.getLogger(__name__)


class PromptBoxDetectionNode(NodeBase):
    """
    Optimized prompt-based box detector.

    Canonical output:
        {
            "tensor": tensor,
            "boxes_xyxy": [(x1, y1, x2, y2), ...],
            "boxes": same as boxes_xyxy for backward compatibility
        }
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_filename: str = self.config.get(
            "model_filename",
            "groundingdino_swinb_cogcoor.pth",
        )
        self.config_path: str = self.config.get(
            "config_path",
            "configs/model_configs/GroundingDINO_SwinB_cfg.py",
        )

        self.prompt: str = self.config.get("prompt", "")
        self.conf: float = float(self.config.get("conf", 0.35))

        self.detect_every_n_frames: int = max(
            1,
            int(self.config.get("detect_every_n_frames", 5)),
        )
        self.reuse_last_boxes: bool = bool(self.config.get("reuse_last_boxes", True))
        self.max_missing_frames: int = int(self.config.get("max_missing_frames", 20))

        # IMPORTANT:
        # If your GroundingDinoWrapper already converts boxes to pixel xyxy,
        # keep this as xyxy_abs.
        #
        # If you directly return raw GroundingDINO boxes from gd_predict(),
        # use cxcywh_norm.
        self.box_format: str = self.config.get("box_format", "xyxy_abs")

        self.min_box_area_ratio: float = float(self.config.get("min_box_area_ratio", 0.0001))
        self.max_box_area_ratio: float = float(self.config.get("max_box_area_ratio", 0.65))
        self.max_boxes: int = int(self.config.get("max_boxes", 50))

        self.model_wrapper: GroundingDinoWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()

        self._frame_idx: int = 0
        self._last_boxes_xyxy: List[Tuple[int, int, int, int]] = []
        self._frames_since_detection: int = 10**9

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name}...")

        resolved_config_path = get_resource_path(Path(self.config_path))

        if not resolved_config_path.exists():
            raise FileNotFoundError(
                f"GroundingDINO config file not found: {resolved_config_path}"
            )

        model_cache_key = f"gdino_{self.model_filename}_{resolved_config_path.name}"

        def model_loader():
            return GroundingDinoWrapper(
                model_filename=self.model_filename,
                config_path=str(resolved_config_path),
            )

        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device),
        )

        self.is_setup = True
        logger.info(f"{self.node_name} setup complete.")

    def reset_state(self) -> None:
        self._frame_idx = 0
        self._last_boxes_xyxy = []
        self._frames_since_detection = 10**9

    def process(self, data: torch.Tensor | Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before setup.")

        if isinstance(data, dict):
            tensor = data.get("tensor")
            result = dict(data)
        else:
            tensor = data
            result = {}

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{self.node_name} expected torch.Tensor, got {type(tensor)}")

        H, W = self._get_hw(tensor)

        if not self.prompt:
            boxes_xyxy = []
            result.update(
                {
                    "tensor": tensor,
                    "boxes_xyxy": boxes_xyxy,
                    "boxes": boxes_xyxy,
                    "frame_idx": self._frame_idx,
                    "prompt": self.prompt,
                }
            )
            self._frame_idx += 1
            return result

        should_detect = (
            self._frame_idx % self.detect_every_n_frames == 0
            or not self._last_boxes_xyxy
        )

        boxes_xyxy: List[Tuple[int, int, int, int]] = []

        if should_detect:
            frame_bgr = self._extract_first_frame_bgr(tensor)

            logger.info(
                f"Running detection for prompt: '{self.prompt}' "
                f"at frame {self._frame_idx}"
            )

            raw_boxes = self.model_wrapper.predict(
                {
                    "image": frame_bgr,
                    "prompt": self.prompt,
                    "conf": self.conf,
                }
            )

            boxes_xyxy = self._convert_and_filter_boxes(raw_boxes, W, H)

            if boxes_xyxy:
                self._last_boxes_xyxy = boxes_xyxy
                self._frames_since_detection = 0
                logger.info(f"Found {len(boxes_xyxy)} usable boxes.")
            else:
                self._frames_since_detection += 1
                logger.info("Found 0 usable boxes.")

        else:
            self._frames_since_detection += 1

        if (
            not boxes_xyxy
            and self.reuse_last_boxes
            and self._last_boxes_xyxy
            and self._frames_since_detection <= self.max_missing_frames
        ):
            boxes_xyxy = self._last_boxes_xyxy

        result.update(
            {
                "tensor": tensor,
                "boxes_xyxy": boxes_xyxy,
                "boxes": boxes_xyxy,  # backward compatibility
                "frame_idx": self._frame_idx,
                "prompt": self.prompt,
            }
        )

        self._frame_idx += 1
        return result

    def _convert_and_filter_boxes(
        self,
        raw_boxes: Any,
        W: int,
        H: int,
    ) -> List[Tuple[int, int, int, int]]:
        if raw_boxes is None:
            return []

        converted = []

        for raw_box in raw_boxes:
            box = convert_box_to_xyxy(
                raw_box,
                width=W,
                height=H,
                box_format=self.box_format,
            )

            if box != (0, 0, 0, 0):
                converted.append(box)

        filtered = filter_boxes_xyxy(
            converted,
            width=W,
            height=H,
            min_area_ratio=self.min_box_area_ratio,
            max_area_ratio=self.max_box_area_ratio,
            max_boxes=self.max_boxes,
        )

        if converted and not filtered:
            logger.warning(
                f"{self.node_name}: all {len(converted)} detected boxes were filtered out. "
                f"Try increasing max_box_area_ratio or lowering conf."
            )

        return filtered

    def _get_hw(self, tensor: torch.Tensor) -> tuple[int, int]:
        if tensor.ndim == 4:
            _, _, H, W = tensor.shape
        elif tensor.ndim == 5:
            _, _, _, H, W = tensor.shape
        else:
            raise ValueError(
                f"{self.node_name} expected 4D or 5D tensor, got {tuple(tensor.shape)}"
            )

        return H, W

    def _extract_first_frame_bgr(self, tensor: torch.Tensor):
        if tensor.ndim == 4:
            frame = tensor[0]
        elif tensor.ndim == 5:
            frame = tensor[0, 0]
        else:
            raise ValueError(
                f"{self.node_name} expected 4D or 5D tensor, got {tuple(tensor.shape)}"
            )

        frame_rgb = (
            frame.detach()
            .float()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")
        self.model_wrapper = None
        self.reset_state()
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")