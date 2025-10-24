import logging
import numpy as np
import torch
import cv2
from typing import Dict, Any

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.realesrgan_wrapper import RealESRGANArchWrapper

logger = logging.getLogger(__name__)


class RealESRGANNode(NodeBase):
    """
    VISUALine pipeline node for Real-ESRGAN upscaling.

    Reads configuration for model filename, scale, tiling, and precision (fp16).
    Uses `RealESRGANArchWrapper` managed by `ResourceManager`. Handles
    RGB <-> BGR conversion for pipeline compatibility.
    """
    use_torch: bool = False

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the node with configuration, reading model details, tile size, and fp16 flag.

        Args:
            config (Dict[str, Any]): Node config, expecting 'model_filename', 'scale',
                                     'tile_size' (optional), 'fp16' (optional).
        """
        super().__init__(config)
        self.model_filename: str = self.config.get("model_filename", "RealESRGAN_x4plus.pth")
        self.scale: int = self.config.get("scale", 4)
        self.tile_size: int = self.config.get("tile_size", 0)  ## default 0 (no tiling)
        self.fp16: bool = self.config.get("fp16", False)  ## default False (use fp32)

        self.model_wrapper: RealESRGANArchWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()
        logger.debug(f"{self.node_name} initialized with config: {self.config}")

    def setup(self, device: torch.device) -> None:
        """
        Requests the `RealESRGANArchWrapper` from the ResourceManager, passing tile and fp16 config.

        Args:
            device (torch.device): The target compute device ('cuda' or 'cpu').
        """
        if self.is_setup:
            logger.debug(f"{self.node_name} is already set up.")
            return

        logger.info(f"Setting up {self.node_name}...")
        
        ## include scale, tile, and fp16 in the cache key for unique configurations
        model_cache_key = f"realesrgan_wrapper_{self.model_filename}_s{self.scale}_t{self.tile_size}_fp16{self.fp16}"

        def model_loader():
            return RealESRGANArchWrapper(
                model_filename=self.model_filename,
                scale=self.scale,
                tile=self.tile_size,
                half=self.fp16
            )
        
        try:
            self.model_wrapper = self._resource_manager.get_model(
                model_name=model_cache_key,
                model_loader=model_loader,
                device=str(device)
            )
            self.is_setup = True
            logger.info(f"{self.node_name} setup complete.")
        except Exception as e:
            logger.critical(f"Failed during {self.node_name} setup: {e}", exc_info=True)
            raise e

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Upscales a batch of RGB NumPy images using the configured wrapper.

        Converts images to BGR for the wrapper, calls predict, converts results back to RGB.

        Args:
            data (np.ndarray): Batch of input images (B, H, W, 3) in RGB format.

        Returns:
            np.ndarray: Batch of upscaled images (B, H_new, W_new, 3) in RGB format.
        """
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")
        if data.ndim != 4 or data.shape[3] != 3:
            raise ValueError(f"Invalid input shape for {self.node_name}: {data.shape}. Expected (B, H, W, 3).")

        ## pre-allocate result list for better memory efficiency
        batch_size = data.shape[0]
        batch_results = []
        batch_results_reserve = batch_size  ## hint for list allocation
        
        try:
            for img_rgb in data:
                ## convert RGB -> BGR for the wrapper (in-place operation where possible)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                ## upscale using the wrapper
                result_bgr = self.model_wrapper.predict(img_bgr)
                
                ## convert BGR -> RGB for the pipeline
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                batch_results.append(result_rgb)
            
            return np.stack(batch_results, axis=0)
            
        except Exception as e:
            logger.error(f"Error during {self.node_name} batch processing: {e}", exc_info=True)
            raise e

    def teardown(self) -> None:
        """Releases the node's reference to the model wrapper. (Called by PipelineManager)."""
        logger.debug(f"Tearing down {self.node_name}...")
        self.model_wrapper = None  ## allow ResourceManager to manage eviction
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")