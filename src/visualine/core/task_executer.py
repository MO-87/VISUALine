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
            device = 'cuda' if data_batch.is_cuda else 'cpu'

            for node in nodes:
                node_uses_torch = getattr(node, "use_torch", True)

                ## if node needs torch but data is on CPU.. move it
                if node_uses_torch and device == 'cpu' and self._use_cuda:
                    data_batch = torch.from_numpy(data_batch).permute(0, 3, 1, 2).float().cuda(non_blocking=True)
                    device = 'cuda'
                
                ## if node needs numpy but data is on GPU.. just move it mooove ittt
                elif not node_uses_torch and device == 'cuda':
                    data_batch = data_batch.permute(0, 2, 3, 1).cpu().numpy()
                    device = 'cpu'

                data_batch = node.process(data_batch)

            ## making sure final output is a torch tensor on the correct device
            if device == 'cpu':
                data_batch = torch.from_numpy(data_batch).permute(0, 3, 1, 2).float()
                if self._use_cuda:
                    data_batch = data_batch.cuda(non_blocking=True)

            return data_batch

        except Exception as e:
            logger.error(f"Error executing batch: {e}", exc_info=True)
            ## always return a tensor on the original device
            if isinstance(data_batch, np.ndarray):
                data_batch = torch.from_numpy(data_batch.astype(np.uint8)).permute(0, 3, 1, 2).float()
                if self._use_cuda:
                    return data_batch.cuda(non_blocking=True)
            return data_batch