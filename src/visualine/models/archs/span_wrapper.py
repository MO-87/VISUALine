import logging
import torch
try:
    import torch_tensorrt
except ImportError:
    pass

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

# Import your patched SPAN architecture here
from visualine.models.archs.span_arch import SPAN 

logger = logging.getLogger(__name__)

class SPANArchWrapper(BaseModelWrapper):
    use_torch: bool = True

    def __init__(self, model_filename: str, scale: int, feature_channels: int = 48, half: bool = False):
        self.model_filename = model_filename
        self.scale = scale
        self.feature_channels = feature_channels
        self.half = half
        self.model: SPAN | None = None
        self._device_str: str = 'cpu'

    def to(self, device: torch.device) -> 'SPANArchWrapper':
        target_device_str = str(device)
        if self.model is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            
            if "cuda" in target_device_str:
                torch.backends.cudnn.benchmark = True

            model_path_str = str(get_model_path(self.model_filename))

            if self.model_filename.endswith(".ts"):
                logger.info(f"Loading optimized Torch-TensorRT model: {self.model_filename}...")
                self.model = torch.jit.load(model_path_str).to(device)
            else:
                logger.info(f"Booting SPAN engine (Scale: {self.scale}x)...")
                
                self.model = SPAN(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    feature_channels=self.feature_channels, 
                    upscale=self.scale
                )
                
                state_dict = torch.load(model_path_str, map_location='cpu')
                
                if 'params_ema' in state_dict:
                    state_dict = state_dict['params_ema']
                elif 'params' in state_dict:
                    state_dict = state_dict['params']
                    
                self.model.load_state_dict(state_dict, strict=True)
                
                self.model.eval()
                self.model = self.model.to(device, memory_format=torch.channels_last)
                
                if self.half:
                    self.model = self.model.half()

                for param in self.model.parameters():
                    param.requires_grad = False

            logger.debug("Running dummy forward pass to fuse kernels...")
            with torch.inference_mode():
                # Note: TRT engine has specific input shape. 
                # We compiled it with 1,3,96,96 to match tile_size=64 + 2*padding=16
                test_shape = (1, 3, 96, 96) if self.model_filename.endswith(".ts") else (1, 3, 64, 64)
                dummy_input = torch.zeros(test_shape, device=device, dtype=torch.float16 if self.half else torch.float32)
                _ = self.model(dummy_input)

            logger.info(f"SPAN Model loaded on {self._device_str}. FP16: {self.half}")
            
        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # TRT compiled with contiguous might need contiguous
        if self.model_filename.endswith(".ts"):
            batch_tensor = batch_tensor.to(memory_format=torch.contiguous_format)
        else:
            batch_tensor = batch_tensor.to(memory_format=torch.channels_last)

        batch_tensor = batch_tensor[:, [2, 1, 0], :, :] / 255.0
        
        if self.half:
            batch_tensor = batch_tensor.half()

        out = self.model(batch_tensor)

        out = torch.clamp(out, 0.0, 1.0)
        out = out[:, [2, 1, 0], :, :] * 255.0

        return out.float().contiguous()

    def cleanup(self) -> None:
        logger.debug(f"Cleaning up resources for SPAN ({self.model_filename})...")
        if self.model is not None:
            try:
                self.model.to('cpu')
            except Exception as e:
                logger.warning(f"Could not move SPAN model to CPU: {e}")
            del self.model
            self.model = None