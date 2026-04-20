import logging
from typing import Any, Dict

import torch

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class GrayscaleNode(NodeBase):
    """A node that converts a batch of frames to grayscale."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("GrayscaleNode initialized.")

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of frames to grayscale.

        Args:
            tensor (torch.Tensor): The input batch as an RGB tensor
                                   of shape (B, 3, H, W).

        Returns:
            torch.Tensor: The batch of grayscaled frames in RGB format
                          of shape (B, 3, H, W).
        """
        ## Vectorized GPU operation applied to the entire batch instantly
        tensor_gray = (
            tensor[:, 0:1, :, :] * 0.299 + 
            tensor[:, 1:2, :, :] * 0.587 + 
            tensor[:, 2:3, :, :] * 0.114
        ).repeat(1, 3, 1, 1)

        return tensor_gray