import logging
from typing import Dict, Any
import torch

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.realesrgan_wrapper import RealESRGANArchWrapper

logger = logging.getLogger(__name__)

class RealESRGANNode(NodeBase):
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_filename: str = self.config.get("model_filename", "RealESRGAN_x4plus.pth")
        self.scale: int = self.config.get("scale", 4)
        self.tile_size: int = self.config.get("tile_size", 512)
        self.tile_pad: int = self.config.get("tile_pad", 10)
        self.fp16: bool = self.config.get("fp16", True)

        self.model_wrapper: RealESRGANArchWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name}...")
        model_cache_key = f"realesrgan_{self.model_filename}_s{self.scale}_t{self.tile_size}_p{self.tile_pad}_fp16{self.fp16}"

        def model_loader():
            return RealESRGANArchWrapper(
                model_filename=self.model_filename,
                scale=self.scale,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                half=self.fp16
            )
        
        self.model_wrapper = self._resource_manager.get_model(
            model_name=model_cache_key,
            model_loader=model_loader,
            device=str(device)
        )
        self.is_setup = True
        logger.info(f"{self.node_name} setup complete.")

    def process(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_setup or self.model_wrapper is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")
        
        is_5d = data.ndim == 5
        if is_5d:
            B, T, C, H, W = data.shape
            data = data.view(B * T, C, H, W)
            
        output = self.model_wrapper.predict(data)
        
        if is_5d:
            output = output.view(B, T, output.shape[1], output.shape[2], output.shape[3])
            
        return output

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")
        self.model_wrapper = None 
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")