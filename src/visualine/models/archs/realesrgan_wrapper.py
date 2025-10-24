import logging
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from visualine.models.loader import get_model_path
from visualine.models.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


class RealESRGANArchWrapper(BaseModelWrapper):
    """
    Wraps the RealESRGAN upscaler for the VISUALine framework.

    Handles loading the model via `ResourceManager` hooks (`to`, `cleanup`)
    and provides the core upscaling logic via the `predict` method, using
    NumPy arrays (BGR format). Accepts tiling and half-precision options.
    """
    use_torch: bool = False

    def __init__(self, model_filename: str, scale: int, tile: int = 0, half: bool = False):
        """
        Initializes the wrapper attributes. Model is loaded later via `to()`.

        Args:
            model_filename (str): Filename of the .pth weights (e.g., "RealESRGAN_x4plus.pth").
            scale (int): Upscale factor of the model (e.g., 4).
            tile (int): Tile size for processing large images. 0 disables tiling. Defaults to 0.
            half (bool): Whether to use float16 (half precision) inference. Defaults to False.
        """
        self.model_filename = model_filename
        self.scale = scale
        self.tile = tile
        self.half = half
        self.upsampler: RealESRGANer | None = None
        self._device_str: str = 'cpu'
        logger.debug(f"RealESRGANArchWrapper for {model_filename} initialized (model not loaded). Config: tile={tile}, half={half}")

        ## store necessary model params - adjust if supporting other models
        ## parameters for RealESRGAN_x4plus.pth
        self.model_params = {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': self.scale
        }

    def to(self, device: torch.device) -> 'RealESRGANArchWrapper':
        """
        Loads the RealESRGAN model onto the specified device. (Called by ResourceManager).

        Instantiates the RRDBNet architecture and then initializes `RealESRGANer`
        with configured tile size and precision.

        Args:
            device (torch.device): The target device (e.g., torch.device('cuda')).

        Returns:
            RealESRGANArchWrapper: self.
        """
        target_device_str = str(device)
        if self.upsampler is None or self._device_str != target_device_str:
            self._device_str = target_device_str
            try:
                model_path_str = str(get_model_path(self.model_filename))

                ## STEP 1: instantiate the actual model architecture first
                if not hasattr(self, 'model_params'):
                    raise AttributeError("model_params not set during init. Cannot create RRDBNet.")

                model_instance = RRDBNet(**self.model_params)
                
                ## enable optimizations for inference
                model_instance.eval()
                for param in model_instance.parameters():
                    param.requires_grad = False
                
                logger.debug("RRDBNet model instance created with inference optimizations.")

                ## STEP 2: initialize RealESRGANer, passing the created model instance
                self.upsampler = RealESRGANer(
                    scale=self.scale,
                    model_path=model_path_str,
                    model=model_instance,
                    tile=self.tile,
                    half=self.half,
                    device=self._device_str
                )

                log_msg = f"RealESRGANer ({self.model_filename}) initialized and loaded onto {self._device_str}"
                if self.tile > 0:
                    log_msg += f" with tiling={self.tile}"
                if self.half:
                    log_msg += " using half precision (fp16)"
                logger.info(log_msg)
                
            except AttributeError as ae:
                logger.critical(f"Initialization error: {ae}", exc_info=True)
                raise ae
            except Exception as e:
                logger.critical(f"Failed to load RealESRGANer model: {e}", exc_info=True)
                self.upsampler = None
                raise e
        elif self.upsampler is not None:
            logger.debug(f"RealESRGANer ({self.model_filename}) already loaded on {self._device_str}.")
        return self

    @torch.inference_mode()
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Upscales a single BGR NumPy image using the loaded RealESRGAN model.

        Args:
            data (np.ndarray): Input image (H, W, 3) in BGR format.

        Returns:
            np.ndarray: Upscaled image (H_new, W_new, 3) in BGR format.
        """
        if self.upsampler is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.predict() called but upsampler is None. "
                "Check for errors during setup/loading."
            )
        try:
            upscaled_img, _ = self.upsampler.enhance(data, outscale=self.scale)
            return upscaled_img
        except Exception as e:
            logger.error(f"Error during RealESRGANer enhancement: {e}", exc_info=True)
            raise e

    def cleanup(self) -> None:
        """
        Cleans up resources when evicted by ResourceManager.

        Moves the underlying torch model to CPU and releases references.
        """
        logger.debug(f"Cleaning up resources for RealESRGANer ({self.model_filename})...")
        
        if self.upsampler and hasattr(self.upsampler, 'model') and self.upsampler.model is not None:
            try:
                self.upsampler.model.to('cpu')
                logger.debug(f"Moved underlying torch model for {self.model_filename} to CPU.")
            except Exception as e:
                logger.warning(f"Could not move RealESRGANer's torch model to CPU during cleanup: {e}")

        if hasattr(self, 'upsampler'):
            del self.upsampler
            self.upsampler = None
        
        ## aggressive cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logger.info(f"Cleaned up and released RealESRGANer ({self.model_filename}).")