import logging
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

class RealESRGANArchWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str, scale: int, tile: int = 0, half: bool = False):
        self.model_filename = model_filename
        self.scale = scale
        self.tile = tile
        self.half = half
        self.upsampler: RealESRGANer | None = None
        self._device_str: str = 'cpu'

        self.model_params = {
            'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64,
            'num_block': 23, 'num_grow_ch': 32, 'scale': self.scale
        }

    def to(self, device: torch.device) -> 'RealESRGANArchWrapper':
        target_device_str = str(device)
        if self.upsampler is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            
            model_path_str = str(get_model_path(self.model_filename))
            model_instance = RRDBNet(**self.model_params)
            
            model_instance.eval()
            for param in model_instance.parameters():
                param.requires_grad = False

            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=model_path_str,
                model=model_instance,
                tile=self.tile,
                half=self.half,
                device=self._device_str
            )
            logger.info(f"RealESRGAN loaded on {self._device_str}. Tile: {self.tile}, FP16: {self.half}")
            
        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        batch_tensor = batch_tensor / 255.0
        if self.half:
            batch_tensor = batch_tensor.half()

        if self.tile == 0:
            out = self.upsampler.model(batch_tensor)
        else:
            outs = []
            for i in range(batch_tensor.size(0)):
                self.upsampler.img = batch_tensor[i].unsqueeze(0)
                self.upsampler.process()
                outs.append(self.upsampler.output.squeeze(0))
            out = torch.stack(outs)

        return (out * 255.0).float()

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