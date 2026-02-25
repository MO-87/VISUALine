import logging
from typing import List, Union, Callable

import numpy as np
import torch
from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)

class TaskExecuter:
    
    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self._use_cuda else "cpu")
        self._execution_plan: List[Callable] = []
        
        if self._use_cuda:
            logger.info(f"TaskExecuter initialized with GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision('high')

    def compile(self, nodes: List[NodeBase], input_format: str = 'torch') -> None:
        logger.info(f"Compiling execution graph for {len(nodes)} nodes...")
        self._execution_plan = []
        current_format = input_format
        
        for node in nodes:
            requires_torch = getattr(node, "use_torch", True)
            
            if requires_torch and current_format == 'numpy':
                self._execution_plan.append(self._optimized_numpy_to_torch)
                current_format = 'torch'
            elif not requires_torch and current_format == 'torch':
                self._execution_plan.append(self._optimized_torch_to_numpy)
                current_format = 'numpy'
                
            self._execution_plan.append(node.process)
            
        logger.debug(f"Compiled plan with {len(self._execution_plan)} steps.")

    def __call__(self, data_batch: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        try:
            for step in self._execution_plan:
                data_batch = step(data_batch)
            return data_batch
            
        except Exception as e:
            logger.error(f"Error executing compiled batch: {e}", exc_info=True)
            raise


    def _optimized_numpy_to_torch(self, arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr)
        
        if self._use_cuda:
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            
        tensor = tensor.permute(0, 3, 1, 2).contiguous().float()        
        return tensor

    def _optimized_torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.dtype != torch.uint8:
            tensor = tensor.byte() 
            
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        return tensor.cpu().numpy()