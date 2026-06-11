import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.span_wrapper import SPANArchWrapper

logger = logging.getLogger(__name__)


class SPANNode(NodeBase):
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_filename: str = self.config.get("model_filename", "spanx4_ch48.pth")
        self.scale: int = int(self.config.get("scale", 4))
        self.feature_channels: int = int(self.config.get("feature_channels", 48))
        self.fp16: bool = bool(self.config.get("fp16", True))

        # tile_size=0 means full-frame inference, fastest if it fits.
        # Use 384/512 for OOM-safe large-video upscale.
        self.tile_size: int = int(self.config.get("tile_size", 0))
        self.tile_overlap: int = int(self.config.get("tile_overlap", 24))

        # If True and input has mask channel, keep/upscale it.
        self.preserve_mask: bool = bool(self.config.get("preserve_mask", False))

        self.model_wrapper: SPANArchWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name}...")

        model_cache_key = (
            f"span_{self.model_filename}"
            f"_s{self.scale}"
            f"_f{self.feature_channels}"
            f"_fp16{self.fp16}"
            f"_tile{self.tile_size}"
            f"_ov{self.tile_overlap}"
        )

        def model_loader():
            return SPANArchWrapper(
                model_filename=self.model_filename,
                scale=self.scale,
                feature_channels=self.feature_channels,
                half=self.fp16,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )

        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device),
        )

        # Do not call self.model_wrapper.to(device) again here.
        # ResourceManager already moves the model.

        self.is_setup = True
        logger.info(f"{self.node_name} setup complete.")

    def process(self, data: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")

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
                f"{self.node_name} expects 4D or 5D tensor, got {tuple(tensor.shape)}"
            )

        channels = batch_like.shape[1]

        if channels not in (3, 4):
            raise ValueError(
                f"{self.node_name} expects 3 or 4 channels, got {channels}. "
                f"Input shape: {tuple(tensor.shape)}"
            )

        rgb = batch_like[:, :3, :, :]
        upscaled_rgb = self.model_wrapper.predict(rgb)

        if channels == 4 and self.preserve_mask:
            mask = batch_like[:, 3:4, :, :]
            target_shape = upscaled_rgb.shape[2:]

            upscaled_mask = F.interpolate(
                mask.float(),
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )

            output = torch.cat([upscaled_rgb, upscaled_mask], dim=1)
        else:
            output = upscaled_rgb

        if restore_shape is not None:
            B, T = restore_shape
            _, C_out, H_out, W_out = output.shape
            output = output.reshape(B, T, C_out, H_out, W_out)

        return output

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")

        self.model_wrapper = None
        self.is_setup = False

        logger.info(f"{self.node_name} teardown complete.")