import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class BokehBlurNode(NodeBase):
    """
    Fast shape-agnostic blur / bokeh node.

    Drop-in replacement for the old BokehBlurNode.

    Supports:
    - 4D RGB:  (B, 3, H, W)
    - 4D RGBM: (B, 4, H, W)
    - 5D RGB:  (B, T, 3, H, W)
    - 5D RGBM: (B, T, 4, H, W)

    Behavior:
    - 3-channel input:
        Applies global blur.

    - 4-channel input:
        Uses channel 4 as a mask.
        mask=1 or 255 means "keep sharp".
        mask=0 means "blur".
        Returns RGB output by default.

    Config:
        blur_intensity: int
        invert_mask: bool
        preserve_mask: bool

    Notes:
    - Uses separable Gaussian blur instead of full 2D convolution.
      This is much faster for large kernels like 51.
    - Output is RGB by default to avoid accidental RGBA/ProRes .mov exports.
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.blur_intensity: int = int(self.config.get("blur_intensity", 51))
        self.invert_mask: bool = bool(self.config.get("invert_mask", False))

        # Default False is important for browser/Streamlit compatibility.
        # If True: returns RGBM when input has mask.
        self.preserve_mask: bool = bool(self.config.get("preserve_mask", False))

        # Ensure kernel size is odd and valid.
        self.blur_intensity = max(3, self.blur_intensity)
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1

        self.mask_kernel_size: int = int(self.config.get("mask_kernel_size", 5))
        self.mask_kernel_size = max(3, self.mask_kernel_size)
        if self.mask_kernel_size % 2 == 0:
            self.mask_kernel_size += 1

        self.blur_kernel_x: torch.Tensor | None = None
        self.blur_kernel_y: torch.Tensor | None = None
        self.mask_kernel_x: torch.Tensor | None = None
        self.mask_kernel_y: torch.Tensor | None = None

        logger.info(
            f"BokehBlurNode initialized. "
            f"Blur intensity: {self.blur_intensity}, "
            f"Invert mask: {self.invert_mask}, "
            f"Preserve mask: {self.preserve_mask}"
        )

    def setup(self, device: torch.device) -> None:
        """
        Pre-compute separable Gaussian kernels once.
        """
        if self.is_setup:
            return

        blur_sigma = max(1.0, self.blur_intensity / 6.0)
        blur_1d = self._get_gaussian_1d(
            size=self.blur_intensity,
            sigma=blur_sigma,
            device=device,
        )

        # RGB depthwise horizontal and vertical kernels.
        self.blur_kernel_x = (
            blur_1d.view(1, 1, 1, self.blur_intensity)
            .repeat(3, 1, 1, 1)
        )

        self.blur_kernel_y = (
            blur_1d.view(1, 1, self.blur_intensity, 1)
            .repeat(3, 1, 1, 1)
        )

        mask_sigma = max(1.0, self.mask_kernel_size / 5.0)
        mask_1d = self._get_gaussian_1d(
            size=self.mask_kernel_size,
            sigma=mask_sigma,
            device=device,
        )

        self.mask_kernel_x = mask_1d.view(1, 1, 1, self.mask_kernel_size)
        self.mask_kernel_y = mask_1d.view(1, 1, self.mask_kernel_size, 1)

        super().setup(device)

    def process(self, data: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            data:
                torch.Tensor:
                    4D: (B, C, H, W)
                    5D: (B, T, C, H, W)

                dict:
                    {"tensor": tensor, ...}

        Returns:
            RGB tensor by default:
                4D: (B, 3, H, W)
                5D: (B, T, 3, H, W)

            If preserve_mask=True and input had 4 channels:
                4D: (B, 4, H, W)
                5D: (B, T, 4, H, W)
        """
        if not self.is_setup:
            raise RuntimeError(f"{self.node_name} process called before setup.")

        if isinstance(data, dict):
            tensor = data.get("tensor")
        else:
            tensor = data

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{self.node_name} expects torch.Tensor, got {type(tensor)}")

        original_ndim = tensor.ndim

        if original_ndim == 4:
            batch_like = tensor
            restore_shape = None

        elif original_ndim == 5:
            B, T, C, H, W = tensor.shape
            batch_like = tensor.reshape(B * T, C, H, W)
            restore_shape = (B, T)

        else:
            raise ValueError(
                f"{self.node_name} expects 4D or 5D tensor, got shape {tuple(tensor.shape)}"
            )

        channels = batch_like.shape[1]

        if channels not in (3, 4):
            raise ValueError(
                f"{self.node_name} expects 3 or 4 channels, got {channels}. "
                f"Input shape: {tuple(tensor.shape)}"
            )

        output = self._process_4d(batch_like)

        if restore_shape is not None:
            B, T = restore_shape
            _, C_out, H_out, W_out = output.shape
            output = output.reshape(B, T, C_out, H_out, W_out)

        return output

    def _process_4d(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process flattened 4D data: (N, C, H, W).
        """
        rgb = data[:, :3].float().clamp(0.0, 255.0)
        has_mask = data.shape[1] == 4

        blurred_rgb = self._blur_rgb(rgb)

        if not has_mask:
            return blurred_rgb

        original_mask = data[:, 3:4].float()

        mask = self._normalize_mask(original_mask)

        if self.invert_mask:
            mask = 1.0 - mask

        smooth_mask = self._blur_mask(mask).clamp(0.0, 1.0)

        # mask=1 -> keep original RGB sharp.
        # mask=0 -> use blurred RGB.
        blended = smooth_mask * rgb + (1.0 - smooth_mask) * blurred_rgb
        blended = blended.clamp(0.0, 255.0)

        if self.preserve_mask:
            return torch.cat([blended, original_mask], dim=1)

        return blended

    def _blur_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Fast separable Gaussian blur for RGB tensors.
        """
        if self.blur_kernel_x is None or self.blur_kernel_y is None:
            raise RuntimeError(f"{self.node_name} blur kernels are not initialized.")

        kernel_x = self.blur_kernel_x.to(device=rgb.device, dtype=rgb.dtype)
        kernel_y = self.blur_kernel_y.to(device=rgb.device, dtype=rgb.dtype)

        pad = self.blur_intensity // 2

        # Replicate padding avoids black borders and is safe for small frames.
        x = F.pad(rgb, (pad, pad, 0, 0), mode="replicate")
        x = F.conv2d(x, kernel_x, groups=3)

        x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        x = F.conv2d(x, kernel_y, groups=3)

        return x

    def _blur_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Small separable Gaussian blur for mask edge softening.
        """
        if self.mask_kernel_x is None or self.mask_kernel_y is None:
            raise RuntimeError(f"{self.node_name} mask kernels are not initialized.")

        kernel_x = self.mask_kernel_x.to(device=mask.device, dtype=mask.dtype)
        kernel_y = self.mask_kernel_y.to(device=mask.device, dtype=mask.dtype)

        pad = self.mask_kernel_size // 2

        x = F.pad(mask, (pad, pad, 0, 0), mode="replicate")
        x = F.conv2d(x, kernel_x)

        x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        x = F.conv2d(x, kernel_y)

        return x

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize mask to 0-1.

        Supports:
        - 0-1 masks
        - 0-255 masks
        """
        if mask.numel() == 0:
            return mask

        # Avoid sync-heavy logic where possible, but this is acceptable per batch.
        if float(mask.detach().max().cpu()) > 1.0:
            mask = mask / 255.0

        return mask.clamp(0.0, 1.0)

    def _get_gaussian_1d(
        self,
        size: int,
        sigma: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate a normalized 1D Gaussian kernel.
        """
        x = torch.arange(
            -size // 2 + 1.0,
            size // 2 + 1.0,
            dtype=torch.float32,
            device=device,
        )

        g = torch.exp(-x.pow(2) / (2.0 * sigma ** 2))
        return g / g.sum()