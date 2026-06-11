import logging
from typing import Optional, Union

import torch

# Try to import torch_tensorrt but allow failure
HAS_TRT = False
try:
    import torch_tensorrt
    HAS_TRT = True
except (ImportError, OSError):
    # Log only once at module level or inside wrapper initialization
    HAS_TRT = False

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper
from visualine.models.archs.span_arch import SPAN

logger = logging.getLogger(__name__)

if not HAS_TRT:
    logger.warning("Torch-TensorRT not found or incompatible. TensorRT features will be disabled.")

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
        self.model: Optional[Union[SPAN, torch.nn.Module]] = None

    def to(self, device: torch.device) -> "SPANArchWrapper":
        target_device = torch.device(device)
        target_device_str = str(target_device)

        if self.model is not None and self._device_str == target_device_str:
            return self

        if target_device.type == "cpu" and self.half:
            logger.warning("CPU device detected. Disabling FP16 for SPAN.")
            self.half = False

        self._device_str = target_device_str

        if target_device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        model_path_str = str(get_model_path(self.model_filename))

        # Check if we should/can load as TRT
        is_trt_file = self.model_filename.endswith(".ts")
        
        if is_trt_file and HAS_TRT:
            logger.info(f"Loading optimized Torch-TensorRT model: {self.model_filename}...")
            self.model = torch.jit.load(model_path_str).to(target_device)
        elif is_trt_file and not HAS_TRT:
            raise RuntimeError(f"Cannot load {self.model_filename}: Torch-TensorRT is not available or incompatible.")
        else:
            logger.info(f"Booting SPAN engine (Scale: {self.scale}x)...")
            self.model = SPAN(
                num_in_ch=3,
                num_out_ch=3,
                feature_channels=self.feature_channels,
                upscale=self.scale,
            )

            state_dict = torch.load(model_path_str, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]

            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            # Use the arch-defined switch_to_deploy if available
            if hasattr(self.model, 'switch_to_deploy'):
                self.model.switch_to_deploy()

            self.model = self.model.to(target_device, memory_format=torch.channels_last)

            if self.half:
                self.model = self.model.half()
            else:
                self.model = self.model.float()

            for param in self.model.parameters():
                param.requires_grad = False

        logger.debug("Running dummy SPAN forward pass to warm up cuDNN...")
        with torch.inference_mode():
            dtype = torch.float16 if self.half else torch.float32
            # Use 96x96 if TRT (static engine), else 64x64
            test_h = 96 if is_trt_file else 64
            dummy_input = torch.zeros(
                1, 3, test_h, test_h,
                device=target_device,
                dtype=dtype,
            )
            if not is_trt_file:
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            
            _ = self.model(dummy_input)

        logger.info(
            f"SPAN Model loaded on {self._device_str}. "
            f"FP16: {self.half}, tile_size={self.tile_size}"
        )

        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if batch_tensor.ndim != 4 or batch_tensor.shape[1] != 3:
            raise ValueError(
                f"SPANArchWrapper expects (B, 3, H, W), got {tuple(batch_tensor.shape)}"
            )

        is_trt_file = self.model_filename.endswith(".ts")

        # VISUALine RGB -> model BGR.
        x = batch_tensor[:, [2, 1, 0], :, :].float().clamp(0.0, 255.0) / 255.0

        if self.half:
            x = x.half()

        if is_trt_file:
            x = x.to(memory_format=torch.contiguous_format)
        else:
            x = x.to(memory_format=torch.channels_last)

        if self.tile_size and self.tile_size > 0:
            out = self._predict_tiled(x)
        else:
            out = self.model(x)

        # SPAN returns BGR in ~0-255 range but without the mean.
        # Restoration: Add BGR mean back
        mean = torch.Tensor([0.4040, 0.4371, 0.4488]).view(1, 3, 1, 1).to(out.device)
        if out.dtype == torch.float16:
            mean = mean.half()
        
        out = out + (mean * 255.0)
        out = torch.clamp(out, 0.0, 255.0)

        # model BGR -> VISUALine RGB.
        out = out[:, [2, 1, 0], :, :]

        return out.float().contiguous()

    def _predict_tiled(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tile = max(64, int(self.tile_size))
        overlap = max(0, int(self.tile_overlap))
        stride = max(1, tile - overlap)
        scale = self.scale
        out_h, out_w = H * scale, W * scale

        output = torch.zeros(B, C, out_h, out_w, dtype=x.dtype, device=x.device)
        weight = torch.zeros_like(output)

        y_positions = self._tile_positions(H, tile, stride)
        x_positions = self._tile_positions(W, tile, stride)

        for y in y_positions:
            for x_pos in x_positions:
                y1, x1 = y, x_pos
                y2, x2 = min(y1 + tile, H), min(x1 + tile, W)
                tile_input = x[:, :, y1:y2, x1:x2]
                tile_output = self.model(tile_input)
                oy1, ox1 = y1 * scale, x1 * scale
                oy2, ox2 = y2 * scale, x2 * scale
                output[:, :, oy1:oy2, ox1:ox2] += tile_output[:, :, : oy2 - oy1, : ox2 - ox1]
                weight[:, :, oy1:oy2, ox1:ox2] += 1.0

        return output / weight.clamp_min(1.0)

    def _tile_positions(self, size: int, tile: int, stride: int) -> list[int]:
        if size <= tile: return [0]
        positions = list(range(0, size - tile + 1, stride))
        if positions[-1] != size - tile: positions.append(size - tile)
        return positions

    def cleanup(self) -> None:
        if getattr(self, "model", None) is not None:
            try: self.model.to("cpu")
            except Exception: pass
            del self.model
            self.model = None
