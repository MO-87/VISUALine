import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torch.amp import autocast

from visualine.core.node_base import NodeBase
from visualine.core.resource_manager import ResourceManager
from visualine.models.loader import get_model_path
from visualine.models.archs.IFNet_HDv3 import IFNet

logger = logging.getLogger(__name__)


class RIFENode(NodeBase):
    """
    VISUALine pipeline node for RIFE frame interpolation..
    it takes B frames and outputs (B * 2) - 1 frames, effectively doubling the framerate
    """
    
    use_torch: bool = True

    @property
    def fps_multiplier(self) -> float:
        """This node effectively doubles the framerate."""
        return 2.0

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_filename: str = self.config.get("model_filename", "flownet.pkl")
        self.scale: float = self.config.get("scale", 1.0)
        self.fp16: bool = self.config.get("fp16", False) 
        
        self.model: IFNet | None = None
        self._resource_manager: ResourceManager = ResourceManager()
        self.last_frame_buffer: torch.Tensor | None = None
        
        logger.debug(f"{self.node_name} initialized with config: {self.config}")

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return

        logger.info(f"Setting up {self.node_name}...")
        self.last_frame_buffer = None
        
        model_cache_key = f"rife_ifnet_{self.model_filename}_fp16{self.fp16}"

        def model_loader():
            model_instance = IFNet()
            model_path = get_model_path(self.model_filename)
            load_net = torch.load(model_path, map_location='cpu')
            weights = {k.replace("module.", ""): v for k, v in load_net.items()}
            
            model_instance.load_state_dict(weights, strict=False) 
            
            model_instance.eval()
            for param in model_instance.parameters():
                param.requires_grad = False
                
            if self.fp16:
                model_instance.half()
            
            model_instance = model_instance.to(memory_format=torch.channels_last)
                
            logger.info(f"IFNet model loaded from {self.model_filename}")
            return model_instance
        
        try:
            self.model = self._resource_manager.get_model(
                model_name=model_cache_key,
                model_loader=model_loader,
                device=str(device)
            )
            self.is_setup = True
            logger.info(f"{self.node_name} setup complete.")
        except Exception as e:
            logger.critical(f"Failed during {self.node_name} setup: {e}", exc_info=True)
            raise e

    @torch.inference_mode()
    def process(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_setup or self.model is None:
            raise RuntimeError(f"{self.node_name} process called before successful setup.")

        calc_dtype = torch.float16 if self.fp16 else torch.float32
        data_norm = data.to(dtype=calc_dtype, copy=True).mul_(1.0 / 255.0)

        if self.last_frame_buffer is None:
            self.last_frame_buffer = data_norm.clone()
            return data.float() if self.fp16 else data

        img0 = self.last_frame_buffer
        img1 = data_norm
        
        self.last_frame_buffer = data_norm.clone()

        _, _, H, W = data.shape
        divisor = max(32, int(64.0 / self.scale))
        pad_h = (divisor - (H % divisor)) % divisor
        pad_w = (divisor - (W % divisor)) % divisor
        
        if pad_h != 0 or pad_w != 0:
            img0 = F.pad(img0, (0, pad_w, 0, pad_h), mode='replicate')
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')

        x = torch.cat((img0, img1), dim=1)
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)

        scale_list = [16.0 / self.scale, 8.0 / self.scale, 4.0 / self.scale, 2.0 / self.scale, 1.0 / self.scale]

        ## run the model!!
        ## use autocast for mixed-precision stability
        with autocast("cuda", enabled=self.fp16):
            interpolated_frame = self.model(x, timestep=0.5, scale_list=scale_list)[2][-1]

        output_batch = torch.cat((interpolated_frame, img1), dim=0)

        final_output = output_batch[..., :H, :W].mul_(255.0)
        
        return final_output.float() if self.fp16 else final_output

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")
        self.model = None 
        self.last_frame_buffer = None 
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")