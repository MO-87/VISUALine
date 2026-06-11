import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from visualine.core.node_base import NodeBase

try:
    from visualine.nodes.utils.box_utils import (
        convert_box_to_xyxy,
        expand_box_xyxy,
        filter_boxes_xyxy,
        get_boxes_from_data,
    )
except Exception:
    convert_box_to_xyxy = None
    expand_box_xyxy = None
    filter_boxes_xyxy = None
    get_boxes_from_data = None

logger = logging.getLogger(__name__)


class SoftBoxMaskNode(NodeBase):
    """
    Converts detected boxes into a 4-channel RGB+Mask tensor.

    Drop-in replacement for the old SoftBoxMaskNode.

    Input:
        torch.Tensor
        OR
        {
            "tensor": tensor,
            "boxes_xyxy": [(x1, y1, x2, y2), ...]
        }

    Backward-compatible input:
        {
            "tensor": tensor,
            "boxes": [...]
        }

    Output:
        RGBM tensor:
        - 4D: (B, 4, H, W)
        - 5D: (B, T, 4, H, W)

    Mask convention:
        mask_value = selected / foreground / keep sharp
        0          = background / blur area

    Typical usage:
        PromptBoxDetectionNode
        -> SoftBoxMaskNode
        -> BokehBlurNode

    Config:
        expand_ratio: float
        default_full_mask: bool
        mask_value: float

        box_format:
            "xyxy_abs" | "xyxy_norm" | "cxcywh_abs" | "cxcywh_norm" | "auto"

        min_box_area_ratio: float
        max_box_area_ratio: float
        max_boxes: int
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.expand_ratio: float = float(self.config.get("expand_ratio", 0.20))

        # If no valid boxes are found:
        # True  -> full mask, so downstream BokehBlur keeps whole frame sharp.
        # False -> zero mask, so downstream BokehBlur blurs whole frame.
        self.default_full_mask: bool = bool(self.config.get("default_full_mask", True))

        self.mask_value: float = float(self.config.get("mask_value", 255.0))

        # Used only if input boxes are not already canonical boxes_xyxy.
        self.box_format: str = str(self.config.get("box_format", "xyxy_abs"))

        # Prevent bad/huge detections from turning the whole frame into a mask.
        self.min_box_area_ratio: float = float(
            self.config.get("min_box_area_ratio", 0.0001)
        )
        self.max_box_area_ratio: float = float(
            self.config.get("max_box_area_ratio", 0.65)
        )
        self.max_boxes: int = int(self.config.get("max_boxes", 20))

        logger.info(
            f"SoftBoxMaskNode initialized. "
            f"expand_ratio={self.expand_ratio}, "
            f"default_full_mask={self.default_full_mask}, "
            f"mask_value={self.mask_value}, "
            f"box_format={self.box_format}, "
            f"min_area={self.min_box_area_ratio}, "
            f"max_area={self.max_box_area_ratio}, "
            f"max_boxes={self.max_boxes}"
        )

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        super().setup(device)

    def process(self, data: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if not self.is_setup:
            raise RuntimeError(f"{self.node_name} process called before setup.")

        if isinstance(data, dict):
            tensor = data.get("tensor")
            raw_boxes = self._get_raw_boxes(data)
        else:
            tensor = data
            raw_boxes = []

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{self.node_name} expected torch.Tensor, got {type(tensor)}"
            )

        if tensor.ndim == 4:
            return self._process_4d(tensor, raw_boxes)

        if tensor.ndim == 5:
            return self._process_5d(tensor, raw_boxes)

        raise ValueError(
            f"{self.node_name} expects 4D or 5D tensor, got {tuple(tensor.shape)}"
        )

    def _process_4d(
        self,
        tensor: torch.Tensor,
        raw_boxes: Iterable[Any],
    ) -> torch.Tensor:
        """
        Process image batch tensor:
            (B, C, H, W) -> (B, 4, H, W)
        """
        B, C, H, W = tensor.shape
        rgb = tensor[:, :3].float().clamp(0.0, 255.0)

        mask = torch.zeros(
            (B, 1, H, W),
            dtype=rgb.dtype,
            device=rgb.device,
        )

        boxes = self._prepare_boxes(
            raw_boxes=raw_boxes,
            width=W,
            height=H,
        )

        if not boxes:
            if self.default_full_mask:
                mask.fill_(self.mask_value)

            return torch.cat([rgb, mask], dim=1)

        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            mask[:, :, y1:y2, x1:x2] = self.mask_value

        return torch.cat([rgb, mask], dim=1)

    def _process_5d(
        self,
        tensor: torch.Tensor,
        raw_boxes: Iterable[Any],
    ) -> torch.Tensor:
        """
        Process video/window tensor:
            (B, T, C, H, W) -> (B, T, 4, H, W)
        """
        B, T, C, H, W = tensor.shape
        rgb = tensor[:, :, :3].float().clamp(0.0, 255.0)

        mask = torch.zeros(
            (B, T, 1, H, W),
            dtype=rgb.dtype,
            device=rgb.device,
        )

        boxes = self._prepare_boxes(
            raw_boxes=raw_boxes,
            width=W,
            height=H,
        )

        if not boxes:
            if self.default_full_mask:
                mask.fill_(self.mask_value)

            return torch.cat([rgb, mask], dim=2)

        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            mask[:, :, :, y1:y2, x1:x2] = self.mask_value

        return torch.cat([rgb, mask], dim=2)

    def _get_raw_boxes(self, data: Dict[str, Any]) -> list:
        """
        Prefer canonical boxes_xyxy. Fallback to older boxes.
        """
        if get_boxes_from_data is not None:
            return get_boxes_from_data(data)

        return data.get("boxes_xyxy", data.get("boxes", []))

    def _prepare_boxes(
        self,
        raw_boxes: Iterable[Any],
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert, expand, and filter boxes.
        """
        if raw_boxes is None:
            return []

        boxes: List[Tuple[int, int, int, int]] = []

        for raw_box in raw_boxes:
            box = self._box_to_xyxy(
                raw_box,
                width=width,
                height=height,
            )

            if box == (0, 0, 0, 0):
                continue

            if expand_box_xyxy is not None:
                box = expand_box_xyxy(
                    box,
                    width=width,
                    height=height,
                    expand_ratio=self.expand_ratio,
                )
            else:
                box = self._fallback_expand_box(
                    box,
                    width=width,
                    height=height,
                    expand_ratio=self.expand_ratio,
                )

            if box != (0, 0, 0, 0):
                boxes.append(box)

        if filter_boxes_xyxy is not None:
            boxes = filter_boxes_xyxy(
                boxes,
                width=width,
                height=height,
                min_area_ratio=self.min_box_area_ratio,
                max_area_ratio=self.max_box_area_ratio,
                max_boxes=self.max_boxes,
            )
        else:
            boxes = self._fallback_filter_boxes(
                boxes,
                width=width,
                height=height,
            )

        return boxes

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

        arr = self._box_to_numpy(raw_box)

        if arr.size < 4:
            return 0, 0, 0, 0

        a, b, c, d = arr[:4]
        fmt = self.box_format.lower().strip()

        if fmt == "auto":
            max_v = float(np.max(np.abs(arr[:4])))

            if max_v <= 1.5:
                # Raw GroundingDINO commonly returns normalized cxcywh.
                fmt = "cxcywh_norm"
            else:
                fmt = "xyxy_abs"

        if fmt == "xyxy_abs":
            x1, y1, x2, y2 = a, b, c, d

        elif fmt == "xyxy_norm":
            x1, y1, x2, y2 = a * width, b * height, c * width, d * height

        elif fmt == "cxcywh_abs":
            cx, cy, bw, bh = a, b, c, d
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

        elif fmt == "cxcywh_norm":
            cx, cy, bw, bh = a * width, b * height, c * width, d * height
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

        else:
            raise ValueError(f"Unsupported box_format: {self.box_format}")

        x1 = int(max(0, min(width - 1, round(float(x1)))))
        y1 = int(max(0, min(height - 1, round(float(y1)))))
        x2 = int(max(0, min(width, round(float(x2)))))
        y2 = int(max(0, min(height, round(float(y2)))))

        if x2 <= x1 or y2 <= y1:
            return 0, 0, 0, 0

        return x1, y1, x2, y2

    def _box_to_numpy(self, box: Any) -> np.ndarray:
        if isinstance(box, torch.Tensor):
            arr = box.detach().cpu().numpy()
        else:
            arr = np.asarray(box)

        return arr.reshape(-1).astype(np.float32)

    def _fallback_expand_box(
        self,
        box: Tuple[int, int, int, int],
        width: int,
        height: int,
        expand_ratio: float,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1

        x1 = int(max(0, min(width - 1, round(x1 - bw * expand_ratio))))
        y1 = int(max(0, min(height - 1, round(y1 - bh * expand_ratio))))
        x2 = int(max(0, min(width, round(x2 + bw * expand_ratio))))
        y2 = int(max(0, min(height, round(y2 + bh * expand_ratio))))

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