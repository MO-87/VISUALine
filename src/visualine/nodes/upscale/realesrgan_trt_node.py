import tensorrt as trt
import torch
import logging
from visualine.core.node_base import NodeBase
from visualine.models.loader import get_model_path

logger = logging.getLogger(__name__)

class RealESRGAN_TRT_Node(NodeBase):
    use_torch = True 

    def __init__(self, config):
        super().__init__(config)
        self.engine_path = config.get("engine_path")
        self.scale = config.get("scale", 4)
        
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        self.input_name = None
        self.output_name = None

    def setup(self, device):
        if self.is_setup:
            return
            
        super().setup(device)
        logger.info(f"Initializing TRT Engine on {device}")
        
        try:
            full_path = str(get_model_path(self.engine_path))
            
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(full_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(device=device)
            
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)
            
            logger.info(f"Setup Complete: {self.input_name} -> {self.output_name}")
        except Exception as e:
            logger.error(f"Critical Setup Failure: {e}")
            self.cleanup()
            raise e

    def process(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_setup or self.context is None:
             raise RuntimeError(f"Node {self.node_name} not properly initialized.")

        batch_input = (batch_tensor.float() / 255.0).contiguous()
        b, c, h, w = batch_input.shape
        
        output_tensor = torch.empty(
            (b, c, h * self.scale, w * self.scale), 
            device=batch_tensor.device, 
            dtype=torch.float32
        ).contiguous()

        with torch.cuda.stream(self.stream):
            self.context.set_input_shape(self.input_name, (b, c, h, w))
            
            self.context.set_tensor_address(self.input_name, batch_input.data_ptr())
            self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())
            
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        self.stream.synchronize() 
        
        return output_tensor.clamp(0, 1) * 255.0

    def cleanup(self):
        logger.info(f"Evicting {self.node_name} from VRAM...")
        self.context = None
        self.engine = None
        self.runtime = None
        if self.stream:
            self.stream.synchronize()
        self.stream = None
        self.is_setup = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"{self.node_name} evicting complete.")

    def teardown(self):
        self.cleanup()