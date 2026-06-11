import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class VideoEnhanceNode(NodeBase):
    """
    Fast local video enhancement node.

    Drop-in replacement for the old VideoEnhanceNode.

    Designed for average consumer GPUs.

    Applies:
    - mild denoising,
    - optional auto-levels,
    - gamma correction,
    - contrast / brightness correction,
    - saturation adjustment,
    - unsharp-mask sharpening.

    Supports:
    - 4D RGB:  (B, 3, H, W)
    - 4D RGBM: (B, 4, H, W)
    - 5D RGB:  (B, T, 3, H, W)
    - 5D RGBM: (B, T, 4, H, W)

    Input/output range:
    - VISUALine standard 0-255.

    Config:
        denoise_strength: float
        sharpen_strength: float
        contrast: float
        brightness: float
        saturation: float
        gamma: float
        kernel_size: int

        auto_levels: bool
        levels_low_percentile: float
        levels_high_percentile: float
        levels_strength: float

        preserve_mask: bool
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.denoise_strength: float = float(self.config.get("denoise_strength", 0.08))
        self.sharpen_strength: float = float(self.config.get("sharpen_strength", 0.30))
        self.contrast: float = float(self.config.get("contrast", 1.08))
        self.brightness: float = float(self.config.get("brightness", 0.00))
        self.saturation: float = float(self.config.get("saturation", 1.08))
        self.gamma: float = float(self.config.get("gamma", 0.95))

        self.kernel_size: int = int(self.config.get("kernel_size", 5))

        # Optional stronger restoration-like correction.
        self.auto_levels: bool = bool(self.config.get("auto_levels", False))
        self.levels_low_percentile: float = float(
            self.config.get("levels_low_percentile", 0.01)
        )
        self.levels_high_percentile: float = float(
            self.config.get("levels_high_percentile", 0.99)
        )
        self.levels_strength: float = float(self.config.get("levels_strength", 0.50))

        # Default False keeps final output RGB and avoids accidental alpha video.
        self.preserve_mask: bool = bool(self.config.get("preserve_mask", False))

        self.denoise_strength = max(0.0, min(1.0, self.denoise_strength))
        self.sharpen_strength = max(0.0, self.sharpen_strength)
        self.saturation = max(0.0, self.saturation)
        self.gamma = max(0.05, self.gamma)
        self.levels_strength = max(0.0, min(1.0, self.levels_strength))

        self.kernel_size = max(3, self.kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.blur_kernel_x: torch.Tensor | None = None
        self.blur_kernel_y: torch.Tensor | None = None
        self.luma_weight: torch.Tensor | None = None

        logger.info(
            f"VideoEnhanceNode initialized. "
            f"denoise={self.denoise_strength}, "
            f"sharpen={self.sharpen_strength}, "
            f"contrast={self.contrast}, "
            f"brightness={self.brightness}, "
            f"saturation={self.saturation}, "
            f"gamma={self.gamma}, "
            f"auto_levels={self.auto_levels}, "
            f"preserve_mask={self.preserve_mask}"
        )

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        sigma = max(1.0, self.kernel_size / 3.0)
        kernel_1d = self._gaussian_1d(self.kernel_size, sigma, device)

        # Separable Gaussian blur kernels.
        # Faster than full 2D conv for larger kernels.
        self.blur_kernel_x = (
            kernel_1d.view(1, 1, 1, self.kernel_size)
            .repeat(3, 1, 1, 1)
        )

        self.blur_kernel_y = (
            kernel_1d.view(1, 1, self.kernel_size, 1)
            .repeat(3, 1, 1, 1)
        )

        self.luma_weight = torch.tensor(
            [0.299, 0.587, 0.114],
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)

        super().setup(device)

    def process(self, data: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if not self.is_setup or self.blur_kernel_x is None or self.blur_kernel_y is None:
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

        mask = data[:, 3:4].float() if has_mask else None

        x = rgb / 255.0

        if self.denoise_strength > 0:
            smooth = self._blur(x)
            x = x * (1.0 - self.denoise_strength) + smooth * self.denoise_strength

        if self.auto_levels:
            leveled = self._auto_levels(x)
            x = x * (1.0 - self.levels_strength) + leveled * self.levels_strength

        if abs(self.gamma - 1.0) > 1e-4:
            x = x.clamp(0.0, 1.0).pow(self.gamma)

        x = (x - 0.5) * self.contrast + 0.5 + self.brightness

        if abs(self.saturation - 1.0) > 1e-4:
            luma = self._luma(x)
            x = luma + self.saturation * (x - luma)

        if self.sharpen_strength > 0:
            smooth = self._blur(x)
            x = x + self.sharpen_strength * (x - smooth)

        rgb_out = x.clamp(0.0, 1.0) * 255.0

        if has_mask and self.preserve_mask:
            return torch.cat([rgb_out, mask], dim=1)

        return rgb_out

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast separable Gaussian blur.
        """
        if self.blur_kernel_x is None or self.blur_kernel_y is None:
            raise RuntimeError(f"{self.node_name} blur kernels are not initialized.")

        kernel_x = self.blur_kernel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.blur_kernel_y.to(device=x.device, dtype=x.dtype)

        pad = self.kernel_size // 2

        x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
        x = F.conv2d(x, kernel_x, groups=3)

        x = F.pad(x, (0, 0, pad, pad), mode="replicate")
        x = F.conv2d(x, kernel_y, groups=3)

        return x

    def _luma(self, x: torch.Tensor) -> torch.Tensor:
        if self.luma_weight is None:
            raise RuntimeError(f"{self.node_name} luma weights are not initialized.")

        luma_weight = self.luma_weight.to(device=x.device, dtype=x.dtype)
        return torch.sum(x * luma_weight, dim=1, keepdim=True)

    def _auto_levels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple per-frame/channel auto-levels.

        x range:
            0-1

        This is optional because quantiles can be a little heavier than the rest
        of the node. Keep auto_levels=False for maximum speed.
        """
        N, C, H, W = x.shape

        flat = x.reshape(N, C, -1)

        low_q = max(0.0, min(0.49, self.levels_low_percentile))
        high_q = max(0.51, min(1.0, self.levels_high_percentile))

        low = torch.quantile(flat, low_q, dim=2, keepdim=True)
        high = torch.quantile(flat, high_q, dim=2, keepdim=True)

        low = low.view(N, C, 1, 1)
        high = high.view(N, C, 1, 1)

        denom = (high - low).clamp_min(1e-4)

        return ((x - low) / denom).clamp(0.0, 1.0)

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