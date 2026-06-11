import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class RemoveBackgroundNode(NodeBase):
    """
    Shape-agnostic node that removes/zeros the background using a mask.

    Drop-in replacement for the old RemoveBackgroundNode.

    Supports:
    - 4D RGB:  (B, 3, H, W)
    - 4D RGBM: (B, 4, H, W)
    - 5D RGB:  (B, T, 3, H, W)
    - 5D RGBM: (B, T, 4, H, W)

    Behavior:
    - 3-channel input:
        No mask exists, so the RGB tensor is returned unchanged.

    - 4-channel input:
        Uses channel 4 as mask.
        mask=1 or 255 means "keep this area".
        mask=0 means "remove / black out this area".

    Config:
        invert_mask: bool
            If False: keep masked subject, remove background.
            If True: remove masked subject, keep background.

        preserve_mask: bool
            If True: returns RGBM when input has mask.
            If False: returns RGB only.

        mask_kernel_size: int
            Smoothing kernel size for softer edges.

        background_value: float
            Background fill value, default 0.0 = black.
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.invert_mask: bool = bool(self.config.get("invert_mask", False))

        # Default True matches your old behavior: return RGB + original mask.
        self.preserve_mask: bool = bool(self.config.get("preserve_mask", True))

        self.background_value: float = float(self.config.get("background_value", 0.0))

        self.mask_kernel_size: int = int(self.config.get("mask_kernel_size", 5))
        self.mask_kernel_size = max(3, self.mask_kernel_size)
        if self.mask_kernel_size % 2 == 0:
            self.mask_kernel_size += 1

        self.mask_kernel_x: torch.Tensor | None = None
        self.mask_kernel_y: torch.Tensor | None = None

        logger.info(
            f"RemoveBackgroundNode initialized. "
            f"Invert mask: {self.invert_mask}, "
            f"Preserve mask: {self.preserve_mask}, "
            f"Background value: {self.background_value}, "
            f"Mask kernel size: {self.mask_kernel_size}"
        )

    def setup(self, device: torch.device) -> None:
        """
        Pre-compute separable Gaussian kernels for mask smoothing.
        """
        if self.is_setup:
            return

        sigma = max(1.0, self.mask_kernel_size / 5.0)
        mask_1d = self._get_gaussian_1d(
            size=self.mask_kernel_size,
            sigma=sigma,
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
            If no mask:
                RGB tensor.

            If mask exists and preserve_mask=True:
                RGBM tensor.

            If mask exists and preserve_mask=False:
                RGB tensor.
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
        Process flattened 4D tensor: (N, C, H, W).
        """
        rgb = data[:, :3].float().clamp(0.0, 255.0)
        has_mask = data.shape[1] == 4

        if not has_mask:
            logger.warning(
                f"{self.node_name} received a 3-channel tensor with no mask. "
                "Cannot remove background. Passing RGB through unchanged."
            )
            return rgb

        original_mask = data[:, 3:4].float()

        mask = self._normalize_mask(original_mask)

        if self.invert_mask:
            mask = 1.0 - mask

        smooth_mask = self._blur_mask(mask).clamp(0.0, 1.0)

        background = torch.full_like(rgb, self.background_value)

        # mask=1 keeps RGB.
        # mask=0 uses background fill.
        isolated_rgb = smooth_mask * rgb + (1.0 - smooth_mask) * background
        isolated_rgb = isolated_rgb.clamp(0.0, 255.0)

        if self.preserve_mask:
            return torch.cat([isolated_rgb, original_mask], dim=1)

        return isolated_rgb

    def _blur_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Separable Gaussian blur for mask edge smoothing.
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