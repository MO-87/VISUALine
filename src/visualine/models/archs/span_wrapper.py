import logging
from typing import Optional

import torch

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper
from visualine.models.archs.span_arch import SPAN

logger = logging.getLogger(__name__)


class SPANArchWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(
        self,
        model_filename: str,
        scale: int,
        feature_channels: int = 48,
        half: bool = False,
        tile_size: int = 0,
        tile_overlap: int = 24,
    ):
        self.model_filename = model_filename
        self.scale = int(scale)
        self.feature_channels = int(feature_channels)
        self.half = bool(half)

        # tile_size=0 means full-frame inference, which is fastest if it fits.
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)

        self._device_str: str = "cpu"

        logger.info(f"Booting SPAN engine (Scale: {self.scale}x)...")

        self.model = SPAN(
            num_in_ch=3,
            num_out_ch=3,
            feature_channels=self.feature_channels,
            upscale=self.scale,
        )

        model_path_str = str(get_model_path(self.model_filename))
        state_dict = torch.load(model_path_str, map_location="cpu")

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        logger.info("Purging training weights and switching SPAN to deploy mode...")
        self.model.switch_to_deploy()

        for param in self.model.parameters():
            param.requires_grad = False

    def to(self, device: torch.device) -> "SPANArchWrapper":
        target_device = torch.device(device)
        target_device_str = str(target_device)

        if target_device.type == "cpu" and self.half:
            logger.warning("CPU device detected. Disabling FP16 for SPAN.")
            self.half = False

        if self._device_str == target_device_str:
            return self

        self._device_str = target_device_str

        if target_device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = self.model.to(target_device, memory_format=torch.channels_last)

        if self.half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        logger.debug("Running dummy SPAN forward pass to warm up cuDNN...")

        with torch.inference_mode():
            dtype = torch.float16 if self.half else torch.float32
            dummy_input = torch.zeros(
                1,
                3,
                64,
                64,
                device=target_device,
                dtype=dtype,
            ).to(memory_format=torch.channels_last)

            _ = self.model(dummy_input)

        logger.info(
            f"SPAN Model loaded on {self._device_str}. "
            f"FP16: {self.half}, tile_size={self.tile_size}"
        )

        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_tensor: RGB tensor, shape (B, 3, H, W), range 0-255.

        Returns:
            RGB tensor, shape (B, 3, H*scale, W*scale), range 0-255.
        """
        if batch_tensor.ndim != 4 or batch_tensor.shape[1] != 3:
            raise ValueError(
                f"SPANArchWrapper expects (B, 3, H, W), got {tuple(batch_tensor.shape)}"
            )

        # VISUALine RGB -> model BGR.
        x = batch_tensor[:, [2, 1, 0], :, :].float().clamp(0.0, 255.0) / 255.0

        if self.half:
            x = x.half()

        x = x.to(memory_format=torch.channels_last)

        if self.tile_size and self.tile_size > 0:
            out = self._predict_tiled(x)
        else:
            out = self.model(x)

        out = torch.clamp(out, 0.0, 1.0)

        # model BGR -> VISUALine RGB.
        out = out[:, [2, 1, 0], :, :] * 255.0

        return out.float().contiguous()

    def _predict_tiled(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tiled inference for large frames.

        Slower than full-frame inference, but much safer for 4x video upscale.
        """
        B, C, H, W = x.shape

        tile = max(64, int(self.tile_size))
        overlap = max(0, int(self.tile_overlap))
        stride = max(1, tile - overlap)

        scale = self.scale

        out_h = H * scale
        out_w = W * scale

        output = torch.zeros(
            B,
            C,
            out_h,
            out_w,
            dtype=x.dtype,
            device=x.device,
        )

        weight = torch.zeros_like(output)

        y_positions = self._tile_positions(H, tile, stride)
        x_positions = self._tile_positions(W, tile, stride)

        for y in y_positions:
            for x_pos in x_positions:
                y1 = y
                x1 = x_pos
                y2 = min(y1 + tile, H)
                x2 = min(x1 + tile, W)

                tile_input = x[:, :, y1:y2, x1:x2]

                tile_output = self.model(tile_input)

                oy1 = y1 * scale
                ox1 = x1 * scale
                oy2 = y2 * scale
                ox2 = x2 * scale

                output[:, :, oy1:oy2, ox1:ox2] += tile_output[
                    :, :, : oy2 - oy1, : ox2 - ox1
                ]

                weight[:, :, oy1:oy2, ox1:ox2] += 1.0

        output = output / weight.clamp_min(1.0)

        return output

    def _tile_positions(self, size: int, tile: int, stride: int) -> list[int]:
        if size <= tile:
            return [0]

        positions = list(range(0, size - tile + 1, stride))

        last = size - tile
        if positions[-1] != last:
            positions.append(last)

        return positions

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for SPAN ({self.model_filename})...")

        if getattr(self, "model", None) is not None:
            try:
                self.model.to("cpu")
            except Exception as e:
                logger.warning(f"Could not move SPAN model to CPU: {e}")

            del self.model
            self.model = None