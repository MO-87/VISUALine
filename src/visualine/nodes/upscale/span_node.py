import logging
from typing import Dict, Any
import torch

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.archs.span_wrapper import SPANArchWrapper

logger = logging.getLogger(__name__)

class SPANNode(NodeBase):
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_filename: str = self.config.get("model_filename", "SPAN_x4.pth")
        self.scale: int = self.config.get("scale", 4)
        self.feature_channels: int = self.config.get("feature_channels", 48)
        self.fp16: bool = self.config.get("fp16", True)

        self.model_wrapper: SPANArchWrapper | None = None
        self._resource_manager: ResourceManager = ResourceManager()

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name}...")
        model_cache_key = f"span_{self.model_filename}_s{self.scale}_f{self.feature_channels}_fp16{self.fp16}"

        def model_loader():
            return SPANArchWrapper(
                model_filename=self.model_filename,
                scale=self.scale,
                feature_channels=self.feature_channels,
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
        
        return self.model_wrapper.predict(data)

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")
        self.model_wrapper = None 
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")