import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.sam2_wrapper import SAM2VideoWrapper

try:
    from visualine.nodes.utils.box_utils import (
        convert_box_to_xyxy,
        filter_boxes_xyxy,
        get_boxes_from_data,
    )
except Exception:
    convert_box_to_xyxy = None
    filter_boxes_xyxy = None
    get_boxes_from_data = None

logger = logging.getLogger(__name__)


class SAM2TrackingNode(NodeBase):
    """
    SAM2 tracking node.

    Expected input:
        {
            "tensor": torch.Tensor,
            "boxes_xyxy": [(x1, y1, x2, y2), ...]
        }

    Backward-compatible input:
        {
            "tensor": torch.Tensor,
            "boxes": [...]
        }

    Output:
        RGB + mask tensor.

        4D image:
            (B, 4, H, W)

        5D video/window:
            (B, T, 4, H, W)

    Mask convention:
        255 = selected / foreground / keep
        0   = unselected / background
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_filename: str = self.config.get(
            "model_filename",
            "sam2_hiera_small.pt",
        )
        self.model_cfg: str = self.config.get(
            "model_cfg",
            "sam2_hiera_s.yaml",
        )
        self.fp16: bool = bool(self.config.get("fp16", True))

        # Used only if this node receives older non-canonical boxes.
        # If input comes from PromptBoxDetectionNode, boxes are already xyxy_abs.
        self.box_format: str = self.config.get("box_format", "xyxy_abs")

        self.min_box_area_ratio: float = float(
            self.config.get("min_box_area_ratio", 0.0001)
        )
        self.max_box_area_ratio: float = float(
            self.config.get("max_box_area_ratio", 0.75)
        )
        self.max_boxes: int = int(self.config.get("max_boxes", 20))

        # If detection fails, output a full mask so downstream selective effects
        # leave the frame visually unchanged instead of changing channel count.
        self.full_mask_when_no_boxes: bool = bool(
            self.config.get("full_mask_when_no_boxes", True)
        )

        self.model_wrapper: SAM2VideoWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name} (SAM2)...")

        model_cache_key = (
            f"sam2_{self.model_filename}_{self.model_cfg}_fp16{self.fp16}"
        )

        def model_loader():
            return SAM2VideoWrapper(
                model_filename=self.model_filename,
                model_cfg=self.model_cfg,
                half=self.fp16,
            )

        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device),
        )

        self.is_setup = True
        logger.info(f"{self.node_name} setup complete.")

    def process(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Takes a dict with:
            - tensor
            - boxes_xyxy or boxes

        Returns:
            RGBM tensor.
        """
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(
                f"{self.node_name} process called before successful setup."
            )

        if not isinstance(data, dict):
            raise TypeError(
                f"{self.node_name} expects dict input, got {type(data)}. "
                "Expected {'tensor': tensor, 'boxes_xyxy': boxes}."
            )

        tensor = data.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{self.node_name} expected data['tensor'] to be torch.Tensor, "
                f"got {type(tensor)}"
            )

        original_is_5d = tensor.ndim == 5

        if tensor.ndim == 4:
            # Image input: (B, C, H, W) -> (B, 1, C, H, W)
            tensor_5d = tensor.unsqueeze(1)
        elif tensor.ndim == 5:
            tensor_5d = tensor
        else:
            raise ValueError(
                f"{self.node_name} expects 4D or 5D tensor, got {tuple(tensor.shape)}"
            )

        B, T, C, H, W = tensor_5d.shape

        if B != 1:
            raise ValueError(
                f"{self.node_name} currently supports batch size 1, got B={B}. "
                "VISUALine video pipeline should run with batch_size=1 for SAM2."
            )

        boxes_xyxy = self._get_valid_boxes_xyxy(data, width=W, height=H)

        if not boxes_xyxy:
            logger.warning(
                f"No valid boxes provided to {self.node_name}. "
                "Returning tensor with full mask."
            )

            out = self._attach_constant_mask(
                tensor_5d,
                value=255.0 if self.full_mask_when_no_boxes else 0.0,
            )

            return out if original_is_5d else out.squeeze(1)

        temp_dir = tempfile.mkdtemp(prefix="visualine_sam2_")
        state = None
        masks_list: List[torch.Tensor] = []

        try:
            logger.debug(f"Extracting {T} frames to temporary directory: {temp_dir}")

            for t in range(T):
                frame_np = (
                    tensor_5d[0, t, :3]
                    .detach()
                    .float()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )

                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(temp_dir, f"{t:05d}.jpg")

                success = cv2.imwrite(frame_path, frame_bgr)
                if not success:
                    raise RuntimeError(f"Failed to write temporary SAM2 frame: {frame_path}")

            state = self.model_wrapper.predict(
                {
                    "action": "init",
                    "video_path": temp_dir,
                }
            )

            for idx, box in enumerate(boxes_xyxy):
                box_np = np.asarray(box, dtype=np.float32).reshape(4)

                self.model_wrapper.predict(
                    {
                        "action": "add_box",
                        "state": state,
                        "frame_idx": 0,
                        "obj_id": idx + 1,
                        "box": box_np,
                    }
                )

            logger.info(
                f"Propagating SAM2 masks through video/window "
                f"with {len(boxes_xyxy)} boxes..."
            )

            for frame_idx, obj_ids, mask_logits in self.model_wrapper.predict(
                {
                    "action": "propagate",
                    "state": state,
                }
            ):
                # Expected SAM2 shape is usually:
                #   [num_objects, 1, H, W]
                # Merge all object masks into one.
                combined_mask = (mask_logits > 0.0).any(dim=0)

                # Normalize to [H, W].
                combined_mask = combined_mask.squeeze()

                if combined_mask.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected SAM2 mask shape after squeeze: "
                        f"{tuple(combined_mask.shape)}"
                    )

                combined_mask = combined_mask.to(
                    dtype=tensor_5d.dtype,
                    device=tensor_5d.device,
                )

                # [H, W] -> [1, H, W], VISUALine mask range 0-255.
                combined_mask = combined_mask.unsqueeze(0) * 255.0

                masks_list.append(combined_mask)

            if not masks_list:
                logger.warning(
                    f"SAM2 produced no masks in {self.node_name}. "
                    "Returning tensor with full mask."
                )

                out = self._attach_constant_mask(
                    tensor_5d,
                    value=255.0 if self.full_mask_when_no_boxes else 0.0,
                )

                return out if original_is_5d else out.squeeze(1)

            # [T, 1, H, W] -> [1, T, 1, H, W]
            mask_tensor = torch.stack(masks_list, dim=0).unsqueeze(0)

            if mask_tensor.shape[1] != tensor_5d.shape[1]:
                logger.warning(
                    f"Mask frame count {mask_tensor.shape[1]} does not match "
                    f"tensor frame count {tensor_5d.shape[1]}. Cropping to shortest length."
                )

                min_t = min(mask_tensor.shape[1], tensor_5d.shape[1])
                tensor_5d = tensor_5d[:, :min_t]
                mask_tensor = mask_tensor[:, :min_t]

            # RGB [1, T, 3, H, W] + mask [1, T, 1, H, W]
            # -> RGBM [1, T, 4, H, W]
            output = torch.cat(
                [
                    tensor_5d[:, :, :3, :, :],
                    mask_tensor,
                ],
                dim=2,
            )

        finally:
            if state is not None:
                try:
                    self.model_wrapper.predict(
                        {
                            "action": "reset",
                            "state": state,
                        }
                    )
                except Exception:
                    logger.warning(
                        f"{self.node_name}: failed to reset SAM2 state.",
                        exc_info=True,
                    )

            shutil.rmtree(temp_dir, ignore_errors=True)

        return output if original_is_5d else output.squeeze(1)

    def _get_valid_boxes_xyxy(
        self,
        data: Dict[str, Any],
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Prefer canonical boxes_xyxy. Fallback to boxes.

        Returns absolute pixel xyxy boxes.
        """
        if get_boxes_from_data is not None:
            raw_boxes = get_boxes_from_data(data)
        else:
            raw_boxes = data.get("boxes_xyxy", data.get("boxes", []))

        if raw_boxes is None:
            return []

        boxes_xyxy: List[Tuple[int, int, int, int]] = []

        for raw_box in raw_boxes:
            box = self._box_to_xyxy(raw_box, width=width, height=height)

            if box != (0, 0, 0, 0):
                boxes_xyxy.append(box)

        if filter_boxes_xyxy is not None:
            boxes_xyxy = filter_boxes_xyxy(
                boxes_xyxy,
                width=width,
                height=height,
                min_area_ratio=self.min_box_area_ratio,
                max_area_ratio=self.max_box_area_ratio,
                max_boxes=self.max_boxes,
            )
        else:
            boxes_xyxy = self._fallback_filter_boxes(
                boxes_xyxy,
                width=width,
                height=height,
            )

        return boxes_xyxy

    def _box_to_xyxy(
        self,
        raw_box: Any,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        if convert_box_to_xyxy is not None:
            return convert_box_to_xyxy(
                raw_box,
                width=width,
                height=height,
                box_format=self.box_format,
            )

        if isinstance(raw_box, torch.Tensor):
            arr = raw_box.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_box)

        arr = arr.reshape(-1).astype(np.float32)

        if arr.size < 4:
            return 0, 0, 0, 0

        x1, y1, x2, y2 = arr[:4]

        if self.box_format == "cxcywh_norm":
            cx = x1 * width
            cy = y1 * height
            bw = x2 * width
            bh = y2 * height

            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

        elif self.box_format == "xyxy_norm":
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        # Default: xyxy_abs.

        x1 = int(max(0, min(width - 1, round(float(x1)))))
        y1 = int(max(0, min(height - 1, round(float(y1)))))
        x2 = int(max(0, min(width, round(float(x2)))))
        y2 = int(max(0, min(height, round(float(y2)))))

        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0, 0

        return x1, y1, x2, y2

    def _fallback_filter_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        frame_area = float(width * height)
        valid: List[Tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area

            if area_ratio < self.min_box_area_ratio:
                continue

            if area_ratio > self.max_box_area_ratio:
                continue

            valid.append((x1, y1, x2, y2))

        return valid[: self.max_boxes]

    def _attach_constant_mask(
        self,
        tensor: torch.Tensor,
        value: float = 255.0,
    ) -> torch.Tensor:
        """
        Attach a constant mask channel so output shape stays consistent.

        Supports:
        - 4D: (B, C, H, W) -> (B, 4, H, W)
        - 5D: (B, T, C, H, W) -> (B, T, 4, H, W)
        """
        if tensor.ndim == 4:
            B, C, H, W = tensor.shape

            mask = torch.full(
                (B, 1, H, W),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            return torch.cat([tensor[:, :3], mask], dim=1)

        if tensor.ndim == 5:
            B, T, C, H, W = tensor.shape

            mask = torch.full(
                (B, T, 1, H, W),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            return torch.cat([tensor[:, :, :3], mask], dim=2)

        raise ValueError(
            f"Unsupported tensor shape for mask attachment: {tuple(tensor.shape)}"
        )

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")

        self.model_wrapper = None
        self.is_setup = False

        logger.info(f"{self.node_name} teardown complete.")