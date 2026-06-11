import logging
from typing import Any, Dict

import torch

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class GrayscaleNode(NodeBase):
    """
    Shape-agnostic grayscale / color-splash node.

    Drop-in replacement for the old GrayscaleNode.

    Supports:
    - 4D RGB:  (B, 3, H, W)
    - 4D RGBM: (B, 4, H, W)
    - 5D RGB:  (B, T, 3, H, W)
    - 5D RGBM: (B, T, 4, H, W)

    Behavior:
    - 3-channel input:
        Applies global grayscale.

    - 4-channel input and background_only=False:
        Applies global grayscale.

    - 4-channel input and background_only=True:
        Uses channel 4 as a mask.
        mask=1 or 255 keeps RGB/color.
        mask=0 converts to grayscale.

    Config:
        background_only: bool
        invert_mask: bool
        preserve_mask: bool

    Notes:
    - Output is RGB by default.
    - preserve_mask=True keeps the 4th mask channel for downstream nodes.
    """

    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.background_only: bool = bool(self.config.get("background_only", False))
        self.invert_mask: bool = bool(self.config.get("invert_mask", False))

        # Default False avoids accidental 4-channel video output.
        self.preserve_mask: bool = bool(self.config.get("preserve_mask", False))

        self.weights: torch.Tensor | None = None

        logger.info(
            f"GrayscaleNode initialized. "
            f"Background only: {self.background_only}, "
            f"Invert mask: {self.invert_mask}, "
            f"Preserve mask: {self.preserve_mask}"
        )

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        self.weights = torch.tensor(
            [0.299, 0.587, 0.114],
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)

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
        if not self.is_setup or self.weights is None:
            raise RuntimeError(f"{self.node_name} process called before setup.")

        if isinstance(data, dict):
            tensor = data.get("tensor")
        else:
            tensor = data

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{self.node_name} expects torch.Tensor, got {type(tensor)}")

        original_ndim = tensor.ndim

        if original_ndim == 4:
            # (B, C, H, W)
            batch_like = tensor
            restore_shape = None

        elif original_ndim == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
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

        gray_3ch = self._to_grayscale_rgb(rgb)

        if not has_mask:
            if self.background_only:
                logger.warning(
                    f"{self.node_name}: background_only=True, but no mask was provided. "
                    "Applying global grayscale."
                )

            return gray_3ch

        original_mask = data[:, 3:4].float()

        if not self.background_only:
            if self.preserve_mask:
                return torch.cat([gray_3ch, original_mask], dim=1)

            return gray_3ch

        mask = self._normalize_mask(original_mask)

        if self.invert_mask:
            mask = 1.0 - mask

        # mask=1 keeps original RGB/color.
        # mask=0 applies grayscale.
        result_rgb = mask * rgb + (1.0 - mask) * gray_3ch
        result_rgb = result_rgb.clamp(0.0, 255.0)

        if self.preserve_mask:
            return torch.cat([result_rgb, original_mask], dim=1)

        return result_rgb

    def _to_grayscale_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to 3-channel grayscale.
        """
        if self.weights is None:
            raise RuntimeError(f"{self.node_name} weights are not initialized.")

        weights = self.weights.to(device=rgb.device, dtype=rgb.dtype)

        gray_1ch = torch.sum(rgb * weights, dim=1, keepdim=True)

        # expand is cheaper than repeat and works fine for downstream ops.
        gray_3ch = gray_1ch.expand(-1, 3, -1, -1)

        return gray_3ch

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