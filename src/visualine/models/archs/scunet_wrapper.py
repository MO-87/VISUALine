import logging
import torch
from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

from visualine.models.archs.network_scunet import SCUNet 

logger = logging.getLogger(__name__)

class SCUNetArchWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str, half: bool = True):
        self.model_filename = model_filename
        self.half = half
        self.model: SCUNet | None = None
        self._device_str: str = 'cpu'

    def to(self, device: torch.device) -> 'SCUNetArchWrapper':
        target_device_str = str(device)
        if self.model is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            
            if "cuda" in target_device_str:
                torch.backends.cudnn.benchmark = True

            logger.info("Booting SCUNet Denoising engine...")
            
            self.model = SCUNet(
                in_nc=3, 
                config=[4, 4, 4, 4, 4, 4, 4], 
                dim=64
            )
            
            model_path_str = str(get_model_path(self.model_filename))
            state_dict = torch.load(model_path_str, map_location='cpu')
            
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
                
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            self.model = self.model.to(device)
            
            if self.half:
                self.model = self.model.half()

            for param in self.model.parameters():
                param.requires_grad = False

            logger.debug("Running dummy forward pass to warm up SCUNet...")
            with torch.inference_mode():
                dummy_input = torch.zeros(1, 3, 64, 64, device=device, dtype=torch.float16 if self.half else torch.float32)
                _ = self.model(dummy_input)

            logger.info(f"SCUNet Model loaded on {self._device_str}. FP16: {self.half}")
            
        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        batch_tensor = batch_tensor / 255.0
        
        if self.half:
            batch_tensor = batch_tensor.half()

        out = self.model(batch_tensor)

        out = torch.clamp(out, 0.0, 1.0)
        out = out * 255.0

        return out.float().contiguous()

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for SCUNet ({self.model_filename})...")
        if self.model is not None:
            try:
                self.model.to('cpu')
            except Exception as e:
                logger.warning(f"Could not move SCUNet model to CPU: {e}")
            del self.model
            self.model = None