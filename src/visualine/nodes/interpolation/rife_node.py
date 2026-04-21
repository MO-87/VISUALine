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
    VISUALine pipeline node for RIFE frame interpolation.
    
    This is a pure PyTorch node (use_torch=True) that processes
    a batch of frames on the GPU. It takes B frames and outputs
    (B * 2) - 1 frames, effectively doubling the framerate.
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

        ## handle fp16 and get original shape
        if self.fp16:
            data = data.half()
        B, C, H, W = data.shape 
        
        ## normalize data to [0, 1] range
        data_norm = data / 255.0

        ## stateful logic
        is_first_batch = False
        if self.last_frame_buffer is None:
            is_first_batch = True
            ## use normalized data
            input_data = data_norm
            self.last_frame_buffer = data_norm[-1:].clone()
        else:
            ## prepend normalized buffer.. very important
            input_data = torch.cat((self.last_frame_buffer, data_norm), dim=0)
            self.last_frame_buffer = data_norm[-1:].clone()

        ## RIFE requires padding.. (a mandatory stepp!!)
        divisor = max(32, int(64.0 / self.scale))
        
        pad_h = (divisor - (H % divisor)) % divisor
        pad_w = (divisor - (W % divisor)) % divisor
        data_padded = F.pad(input_data, (0, pad_w, 0, pad_h), mode='replicate')

        ## create input batches
        img0 = data_padded[:-1]
        img1 = data_padded[1:]
        x = torch.cat((img0, img1), 1)

        ## define scale list
        scale_list = [16/self.scale, 8/self.scale, 4/self.scale, 2/self.scale, 1/self.scale]

        ## run the model!!
        ## use autocast for mixed-precision stability
        with autocast("cuda", enabled=self.fp16):
            _, _, merged = self.model(x, timestep=0.5, scale_list=scale_list)
        
        ## merged contains the interpolated frames
        interpolated_frames = merged[-1]

        ## reconstruct the full framerate batch
        if is_first_batch:
            ## interleave original(0 to B-1) and interpolated
            ## result: [Orig0, Interp0, Orig1, Interp1, ... OrigB-1]
            original_frames = data_padded
            output_stack = torch.stack((original_frames[:-1], interpolated_frames), dim=1)
            output_batch = output_stack.flatten(0, 1)
            ## add the very last original frame to close the batch
            output_batch = torch.cat((output_batch, original_frames[-1:]), dim=0)
        else:
            ## buffer already handled the first frame, so interleave Interp and Orig
            ## result: [Interp0, Orig1, Interp1, Orig2, ... OrigB]
            original_frames_new = data_padded[1:]
            output_stack = torch.stack((interpolated_frames, original_frames_new), dim=1)
            output_batch = output_stack.flatten(0, 1)

        ## un-pad the result
        final_output_norm = output_batch[:, :, :H, :W]

        ## de-normalize data back to [0, 255] range
        final_output = final_output_norm * 255.0
        
        ## return as fp32 for compatibility
        if self.fp16:
            final_output = final_output.float()
            
        return final_output

    def teardown(self) -> None:
        logger.debug(f"Tearing down {self.node_name}...")
        self.model = None 
        self.last_frame_buffer = None 
        self.is_setup = False
        logger.info(f"{self.node_name} teardown complete.")