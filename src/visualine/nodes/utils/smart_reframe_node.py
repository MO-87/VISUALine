import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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


class SmartReframeNode(NodeBase):
    """
    Converts videos/images to a target aspect ratio while following detected boxes.

    Drop-in replacement for the old SmartReframeNode.

    Typical use:
    - landscape video -> vertical 9:16 short
    - horizontal input -> centered subject crop

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
        RGB tensor with configured output size.

    Supports:
    - 4D: (B, C, H, W)
    - 5D: (B, T, C, H, W)

    Config:
        output_width: int
        output_height: int
        smoothing: float
        box_expansion: float
        box_format: "xyxy_abs" | "xyxy_norm" | "cxcywh_abs" | "cxcywh_norm" | "auto"
        min_box_area_ratio: float
        max_box_area_ratio: float
        max_boxes: int
        fallback: "center" | "keep_last"
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.output_width: int = int(self.config.get("output_width", 720))
        self.output_height: int = int(self.config.get("output_height", 1280))

        self.output_width = max(2, self.output_width)
        self.output_height = max(2, self.output_height)

        self.smoothing: float = float(self.config.get("smoothing", 0.85))
        self.smoothing = max(0.0, min(0.99, self.smoothing))

        self.box_expansion: float = float(self.config.get("box_expansion", 0.25))

        # Used only when boxes are not already canonical boxes_xyxy.
        self.box_format: str = str(self.config.get("box_format", "xyxy_abs"))

        self.min_box_area_ratio: float = float(
            self.config.get("min_box_area_ratio", 0.0001)
        )
        self.max_box_area_ratio: float = float(
            self.config.get("max_box_area_ratio", 0.85)
        )
        self.max_boxes: int = int(self.config.get("max_boxes", 20))

        # center:
        #   if no boxes, crop center.
        #
        # keep_last:
        #   if no boxes but previous center exists, keep following last known center.
        self.fallback: str = str(self.config.get("fallback", "keep_last")).lower().strip()

        if self.fallback not in {"center", "keep_last"}:
            logger.warning(
                f"{self.node_name}: Unsupported fallback '{self.fallback}'. "
                "Using 'keep_last'."
            )
            self.fallback = "keep_last"

        self._center_x: float | None = None
        self._center_y: float | None = None

        logger.info(
            f"SmartReframeNode initialized. "
            f"output={self.output_width}x{self.output_height}, "
            f"smoothing={self.smoothing}, "
            f"box_expansion={self.box_expansion}, "
            f"box_format={self.box_format}, "
            f"fallback={self.fallback}"
        )

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        super().setup(device)

    def reset_state(self) -> None:
        self._center_x = None
        self._center_y = None

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

        original_ndim = tensor.ndim

        if original_ndim == 4:
            # (B, C, H, W)
            batch_like = tensor[:, :3]
            restore_shape = None

        elif original_ndim == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            B, T, C, H, W = tensor.shape
            batch_like = tensor[:, :, :3].reshape(B * T, 3, H, W)
            restore_shape = (B, T)

        else:
            raise ValueError(
                f"{self.node_name} expects 4D or 5D tensor, got {tuple(tensor.shape)}"
            )

        output = self._process_4d(batch_like, raw_boxes)

        if restore_shape is not None:
            B, T = restore_shape
            _, C_out, H_out, W_out = output.shape
            output = output.reshape(B, T, C_out, H_out, W_out)

        return output

    def _process_4d(
        self,
        rgb: torch.Tensor,
        raw_boxes: Iterable[Any],
    ) -> torch.Tensor:
        """
        Process flattened 4D RGB tensor: (N, 3, H, W).
        """
        rgb = rgb[:, :3].float().clamp(0.0, 255.0)

        N, C, H, W = rgb.shape

        boxes = self._prepare_boxes(raw_boxes, width=W, height=H)

        frames = []

        for i in range(N):
            frame = rgb[i : i + 1]

            target_cx, target_cy = self._target_center(
                boxes=boxes,
                width=W,
                height=H,
            )

            if self._center_x is None or self._center_y is None:
                self._center_x = target_cx
                self._center_y = target_cy
            else:
                self._center_x = (
                    self.smoothing * self._center_x
                    + (1.0 - self.smoothing) * target_cx
                )
                self._center_y = (
                    self.smoothing * self._center_y
                    + (1.0 - self.smoothing) * target_cy
                )

            crop = self._crop_to_aspect(
                frame=frame,
                cx=self._center_x,
                cy=self._center_y,
            )

            resized = F.interpolate(
                crop,
                size=(self.output_height, self.output_width),
                mode="bilinear",
                align_corners=False,
            )

            frames.append(resized)

        return torch.cat(frames, dim=0)

    def _target_center(
        self,
        boxes: List[Tuple[int, int, int, int]],
        width: int,
        height: int,
    ) -> Tuple[float, float]:
        """
        Compute the target crop center from boxes.

        If no boxes:
        - fallback='keep_last' and previous center exists: keep previous center
        - otherwise: image center
        """
        if not boxes:
            if (
                self.fallback == "keep_last"
                and self._center_x is not None
                and self._center_y is not None
            ):
                return self._center_x, self._center_y

            return width / 2.0, height / 2.0

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _crop_to_aspect(
        self,
        frame: torch.Tensor,
        cx: float,
        cy: float,
    ) -> torch.Tensor:
        """
        Crop frame around center to match target output aspect ratio.
        """
        _, _, H, W = frame.shape

        target_aspect = self.output_width / self.output_height
        input_aspect = W / H

        if input_aspect > target_aspect:
            # Input is wider than target: crop width.
            crop_h = H
            crop_w = int(round(H * target_aspect))
        else:
            # Input is taller/narrower than target: crop height.
            crop_w = W
            crop_h = int(round(W / target_aspect))

        crop_w = max(2, min(W, crop_w))
        crop_h = max(2, min(H, crop_h))

        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))

        x1 = max(0, min(W - crop_w, x1))
        y1 = max(0, min(H - crop_h, y1))

        return frame[:, :, y1 : y1 + crop_h, x1 : x1 + crop_w]

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
                    expand_ratio=self.box_expansion,
                )
            else:
                box = self._fallback_expand_box(
                    box,
                    width=width,
                    height=height,
                    expand_ratio=self.box_expansion,
                )

            if box != (0, 0, 0, 0):
                boxes.append(box)

        if filter_boxes_xyxy is not None:
            return filter_boxes_xyxy(
                boxes,
                width=width,
                height=height,
                min_area_ratio=self.min_box_area_ratio,
                max_area_ratio=self.max_box_area_ratio,
                max_boxes=self.max_boxes,
            )

        return self._fallback_filter_boxes(
            boxes,
            width=width,
            height=height,
        )

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