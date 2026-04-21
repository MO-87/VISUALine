import logging
import torch
import torch.nn.functional as F
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

class RealESRGANArchWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str, scale: int, tile: int = 0, tile_pad: int = 10, half: bool = False):
        self.model_filename = model_filename
        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.half = half
        self.upsampler: RealESRGANer | None = None
        self._device_str: str = 'cpu'

        self.rrdb_params = {
            'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64,
            'num_block': 23, 'num_grow_ch': 32, 'scale': self.scale
        }
        
        self.srvgg_params = {
            'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64,
            'num_conv': 16, 'upscale': self.scale, 'act_type': 'prelu'
        }

    def to(self, device: torch.device) -> 'RealESRGANArchWrapper':
        target_device_str = str(device)
        if self.upsampler is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            
            # 1. Enable CuDNN Benchmarking for faster convolution selection
            if "cuda" in target_device_str:
                torch.backends.cudnn.benchmark = True
            
            model_path_str = str(get_model_path(self.model_filename))
            filename_lower = self.model_filename.lower()
            
            if "animevideo" in filename_lower or "compact" in filename_lower:
                logger.info("Fast Video Model detected. Booting SRVGGNetCompact engine...")
                model_instance = SRVGGNetCompact(**self.srvgg_params)
            else:
                logger.info("Heavy Photo Model detected. Booting RRDBNet engine...")
                model_instance = RRDBNet(**self.rrdb_params)
            
            # 2. Push model to device and convert to channels_last memory format
            model_instance = model_instance.to(device, memory_format=torch.channels_last)
            
            if self.half:
                model_instance = model_instance.half()

            model_instance.eval()
            for param in model_instance.parameters():
                param.requires_grad = False

            # 3. Initialize RealESRGANer FIRST (This safely loads the state_dict)
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=model_path_str,
                model=model_instance,
                tile=self.tile,
                tile_pad=self.tile_pad,
                half=self.half,
                device=self._device_str
            )

            logger.info(f"Model loaded on {self._device_str}. Tile: {self.tile}, Pad: {self.tile_pad}, FP16: {self.half}")
            
        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # Convert input to channels_last to match the optimized model weights
        batch_tensor = batch_tensor.to(memory_format=torch.channels_last)

        # Permute RGB to BGR and normalize in one go
        batch_tensor = batch_tensor[:, [2, 1, 0], :, :] / 255.0
        
        if self.half:
            batch_tensor = batch_tensor.half()

        if self.tile == 0:
            out = self.upsampler.model(batch_tensor)
        else:
            # Note: Tiling in Python is a bottleneck. 
            # If batch sizes are large, consider optimizing the upstream realesrganer tiling logic to batch crops.
            outs = []
            for i in range(batch_tensor.size(0)):
                self.upsampler.img = batch_tensor[i].unsqueeze(0)
                self.upsampler.process() 
                outs.append(self.upsampler.output.squeeze(0))
            out = torch.stack(outs)

        # Output comes back. Clamp, permute BGR back to RGB, scale back to 255
        out = torch.clamp(out, 0.0, 1.0)
        out = out[:, [2, 1, 0], :, :] * 255.0

        # Ensure contiguous memory format before returning to the rest of the pipeline
        return out.float().contiguous()

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for RealESRGANer ({self.model_filename})...")
        if self.upsampler and hasattr(self.upsampler, 'model') and self.upsampler.model is not None:
            try:
                self.upsampler.model.to('cpu')
            except Exception as e:
                logger.warning(f"Could not move torch model to CPU: {e}")

        if hasattr(self, 'upsampler'):
            del self.upsampler
            self.upsampler = None