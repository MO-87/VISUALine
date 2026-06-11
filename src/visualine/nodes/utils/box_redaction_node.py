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


class BoxRedactionNode(NodeBase):
    """
    Fast privacy redaction using detected boxes.

    Drop-in replacement for the old BoxRedactionNode.

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
        RGB tensor with boxed regions blurred, pixelated, or covered.

    Supports:
    - 4D: (B, C, H, W)
    - 5D: (B, T, C, H, W)

    Config:
        mode: "blur" | "pixelate" | "black" | "white"
        blur_intensity: int
        pixel_size: int
        expand_ratio: float
        box_format: "xyxy_abs" | "xyxy_norm" | "cxcywh_abs" | "cxcywh_norm" | "auto"
        min_box_area_ratio: float
        max_box_area_ratio: float
        max_boxes: int
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.mode: str = str(self.config.get("mode", "blur")).lower().strip()

        if self.mode not in {"blur", "pixelate", "black", "white"}:
            logger.warning(
                f"{self.node_name}: Unsupported mode '{self.mode}'. Falling back to 'blur'."
            )
            self.mode = "blur"

        self.blur_intensity: int = int(self.config.get("blur_intensity", 51))
        self.pixel_size: int = max(2, int(self.config.get("pixel_size", 16)))
        self.expand_ratio: float = float(self.config.get("expand_ratio", 0.08))

        # Used only if boxes are not already canonical boxes_xyxy.
        self.box_format: str = str(self.config.get("box_format", "xyxy_abs"))

        # Prevent accidental full-frame redaction from bad/too-broad detections.
        self.min_box_area_ratio: float = float(
            self.config.get("min_box_area_ratio", 0.00005)
        )
        self.max_box_area_ratio: float = float(
            self.config.get("max_box_area_ratio", 0.65)
        )
        self.max_boxes: int = int(self.config.get("max_boxes", 80))

        self.blur_intensity = max(3, self.blur_intensity)
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1

        self.blur_kernel_x: torch.Tensor | None = None
        self.blur_kernel_y: torch.Tensor | None = None

        logger.info(
            f"BoxRedactionNode initialized. "
            f"mode={self.mode}, blur_intensity={self.blur_intensity}, "
            f"pixel_size={self.pixel_size}, expand_ratio={self.expand_ratio}, "
            f"box_format={self.box_format}, "
            f"min_area={self.min_box_area_ratio}, "
            f"max_area={self.max_box_area_ratio}, "
            f"max_boxes={self.max_boxes}"
        )

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        sigma = max(1.0, self.blur_intensity / 6.0)
        kernel_1d = self._gaussian_1d(self.blur_intensity, sigma, device)

        # Separable Gaussian kernels.
        # Much faster than full 2D convolution for large kernels.
        self.blur_kernel_x = (
            kernel_1d.view(1, 1, 1, self.blur_intensity)
            .repeat(3, 1, 1, 1)
        )

        self.blur_kernel_y = (
            kernel_1d.view(1, 1, self.blur_intensity, 1)
            .repeat(3, 1, 1, 1)
        )

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
        output = rgb.clone()

        boxes = self._prepare_boxes(raw_boxes, width=W, height=H)

        if not boxes:
            return output

        if self.mode == "pixelate":
            return self._apply_pixelate(output, boxes)

        if self.mode == "black":
            return self._apply_fill(output, boxes, value=0.0)

        if self.mode == "white":
            return self._apply_fill(output, boxes, value=255.0)

        # Default: blur.
        return self._apply_blur(output, boxes)

    def _apply_blur(
        self,
        output: torch.Tensor,
        boxes: List[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """
        Blur the whole frame once, then copy only redaction regions.
        This is faster than blurring each region separately.
        """
        blurred = self._blur_rgb(output)

        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            output[:, :, y1:y2, x1:x2] = blurred[:, :, y1:y2, x1:x2]

        return output

    def _apply_pixelate(
        self,
        output: torch.Tensor,
        boxes: List[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            region = output[:, :, y1:y2, x1:x2]

            if region.numel() == 0:
                continue

            rh, rw = region.shape[-2:]

            small_h = max(1, rh // self.pixel_size)
            small_w = max(1, rw // self.pixel_size)

            small = F.interpolate(
                region,
                size=(small_h, small_w),
                mode="nearest",
            )

            pixelated = F.interpolate(
                small,
                size=(rh, rw),
                mode="nearest",
            )

            output[:, :, y1:y2, x1:x2] = pixelated

        return output

    def _apply_fill(
        self,
        output: torch.Tensor,
        boxes: List[Tuple[int, int, int, int]],
        value: float,
    ) -> torch.Tensor:
        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue

            output[:, :, y1:y2, x1:x2] = value

        return output

    def _blur_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Fast separable Gaussian blur.
        """
        if self.blur_kernel_x is None or self.blur_kernel_y is None:
            raise RuntimeError(f"{self.node_name} blur kernels are not initialized.")

        kernel_x = self.blur_kernel_x.to(device=rgb.device, dtype=rgb.dtype)
        kernel_y = self.blur_kernel_y.to(device=rgb.device, dtype=rgb.dtype)

        pad = self.blur_intensity // 2

        # Replicate padding avoids border artifacts and is safer than reflect for small frames.
        x = F.pad(rgb, (pad, pad, 0, 0), mode="replicate")
        x = F.conv2d(x, kernel_x, groups=3)

        x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        x = F.conv2d(x, kernel_y, groups=3)

        return x

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

    def _gaussian_1d(
        self,
        size: int,
        sigma: float,
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.arange(
            -size // 2 + 1.0,
            size // 2 + 1.0,
            dtype=torch.float32,
            device=device,
        )

        g = torch.exp(-x.pow(2) / (2.0 * sigma**2))
        return g / g.sum()