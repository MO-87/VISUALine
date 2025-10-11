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
            for node in nodes:
                ## handle torch-based nodes
                if getattr(node, "use_torch", False):
                    data_batch = node.process(data_batch)
                    continue

                ## handle numpy-based nodes
                logger.debug(f"Converting batch for CPU node: {node.node_name}")
                batch_np = data_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                processed_np = np.stack([node.process(f) for f in batch_np])
                data_batch = torch.from_numpy(processed_np).permute(0, 3, 1, 2).float()

                ## move back to GPU if necessary
                if self._use_cuda:
                    data_batch = data_batch.cuda(non_blocking=True)

            return data_batch

        except Exception as e:
            logger.error(f"Error executing batch: {e}", exc_info=True)
            return data_batch

    def execute_frame(self, node: NodeBase, frame: Any) -> Any:
        """
        Executes a single frame through one node (used for fallback mode).
        """
        try:
            ## handle PyTorch-based node
            if getattr(node, "use_torch", False):
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
                if self._use_cuda:
                    frame_tensor = frame_tensor.cuda(non_blocking=True)
                result = node.process(frame_tensor)
                return result.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

            ## otherwise, assume OpenCV-based node
            return node.process(frame)

        except Exception as e:
            logger.error(f"Error executing node '{node.__class__.__name__}': {e}")
            return frame
