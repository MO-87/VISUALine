import logging
from typing import List, Callable

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

    def compile(self, nodes: List[NodeBase]) -> None:
        logger.info(f"Compiling execution graph for {len(nodes)} nodes...")
        self._execution_plan = [node.process for node in nodes]
        logger.debug(f"Compiled plan with {len(self._execution_plan)} steps.")

    @torch.inference_mode()
    def __call__(self, data_batch: torch.Tensor) -> torch.Tensor:
        try:
            for step in self._execution_plan:
                data_batch = step(data_batch)
            return data_batch
            
        except Exception as e:
            logger.error(f"Error executing compiled batch: {e}", exc_info=True)
            raise