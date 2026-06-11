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
    """
    A wrapper around the Real-ESRGAN model for inference.
    """

    use_torch: bool = True

    def __init__(self, model_filename: str, scale: int, tile: int = 0, half: bool = False):
        self.model_filename = model_filename
        self.scale = scale
        self.tile = tile
        self.half = half
        self._device_str = 'cpu'
        self.upsampler = None

    def to(self, device: torch.device) -> 'RealESRGANArchWrapper':
        target_device_str = str(device)
        if self.upsampler is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            
            if "cuda" in target_device_str:
                torch.backends.cudnn.benchmark = True

            model_path = str(get_model_path(self.model_filename))
            
            if self.model_filename.endswith(".ts"):
                # Load pre-compiled TorchScript/TensorRT model
                self.model = torch.jit.load(model_path).to(device)
                self.model.eval()
                logger.info(f"Loaded compiled RealESRGAN model from {self.model_filename}")
                return self

            # Select architecture based on model name
            if 'anime' in self.model_filename.lower() or 'realesr-v3' in self.model_filename.lower():
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=self.scale)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)

            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                model=model,
                tile=self.tile,
                tile_pad=10,
                pre_pad=0,
                half=self.half,
                device=device
            )
            logger.info(f"RealESRGAN Model loaded on {self._device_str}. FP16: {self.half}")
            
        return self

    @torch.inference_mode()
    def predict(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_tensor: RGB tensor (B, C, H, W) 0-255.
        Returns:
            RGB tensor (B, C, H*scale, W*scale) 0-255.
        """
        if self.model_filename.endswith(".ts"):
            # VISUALine RGB -> model BGR [0, 1]
            x = batch_tensor[:, [2, 1, 0], :, :].float().clamp(0.0, 255.0) / 255.0
            if self.half: x = x.half()
            
            out = self.model(x)
            
            # model BGR -> VISUALine RGB [0, 255]
            out = torch.clamp(out, 0.0, 1.0)
            out = out[:, [2, 1, 0], :, :] * 255.0
            return out.float().contiguous()

        # For .pth models, we use RealESRGANer wrapper which expects numpy BGR
        results = []
        for i in range(batch_tensor.shape[0]):
            # Convert single frame to numpy BGR (MUST BE UINT8 for RealESRGANer)
            img_rgb = batch_tensor[i].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype('uint8')
            img_bgr = img_rgb[:, :, ::-1]
            
            # Enhance (returns numpy BGR uint8)
            output, _ = self.upsampler.enhance(img_bgr, outscale=self.scale)
            
            # Convert back to RGB Tensor
            out_rgb = output[:, :, ::-1].copy()
            results.append(torch.from_numpy(out_rgb).permute(2, 0, 1))
            
        return torch.stack(results).float().to(batch_tensor.device)

    def cleanup(self) -> None:
        if self.upsampler is not None:
            try:
                self.upsampler.model.to('cpu')
            except Exception as e:
                logger.warning(f"Could not move RealESRGAN model to CPU: {e}")

        if hasattr(self, 'upsampler'):
            del self.upsampler
            self.upsampler = None
