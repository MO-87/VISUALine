import logging
from typing import Any, List

import numpy as np
import torch
from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class TaskExecuter:
    """Executes a node (or series of nodes) efficiently on a batch of data."""
    
    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            logger.info(f"TaskExecuter initialized with GPU: {torch.cuda.get_device_name(0)}")
            ## enable cudnn benchmarking for faster convolutions
            torch.backends.cudnn.benchmark = True
        else:
            logger.info("TaskExecuter initialized on CPU.")

    def execute_batch(self, nodes: List[NodeBase], data_batch: torch.Tensor) -> torch.Tensor:
        """
        Executes a batch of data sequentially through all given nodes.
        
        Args:
            nodes (List[NodeBase]): The nodes to execute in order.
            data_batch (torch.Tensor): Batch of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: The processed batch (on same device as input).
        """
        try:
            is_cuda = data_batch.is_cuda
            current_format = 'torch'  ## track current data format
            
            for node in nodes:
                node_uses_torch = getattr(node, "use_torch", True)
                
                ## convert torch -> numpy if needed
                if not node_uses_torch and current_format == 'torch':
                    data_batch = data_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    current_format = 'numpy'
                
                ## convert numpy -> torch if needed
                elif node_uses_torch and current_format == 'numpy':
                    data_batch = torch.from_numpy(data_batch).permute(0, 3, 1, 2).float()
                    if self._use_cuda:
                        data_batch = data_batch.pin_memory().cuda(non_blocking=True)
                    current_format = 'torch'
                
                ## process through node
                data_batch = node.process(data_batch)
            
            ## ensure final output is torch tensor on correct device
            if current_format == 'numpy':
                data_batch = torch.from_numpy(data_batch).permute(0, 3, 1, 2).float()
                if self._use_cuda:
                    data_batch = data_batch.pin_memory().cuda(non_blocking=True)
            
            ## ensure output matches input device
            if is_cuda and not data_batch.is_cuda:
                data_batch = data_batch.cuda(non_blocking=True)
            elif not is_cuda and data_batch.is_cuda:
                data_batch = data_batch.cpu()
            
            return data_batch
            
        except Exception as e:
            logger.error(f"Error executing batch: {e}", exc_info=True)
            ## attempt recovery by returning tensor on original device
            if isinstance(data_batch, np.ndarray):
                data_batch = torch.from_numpy(data_batch.astype(np.uint8)).permute(0, 3, 1, 2).float()
                if self._use_cuda and is_cuda:
                    data_batch = data_batch.cuda(non_blocking=True)
            return data_batch