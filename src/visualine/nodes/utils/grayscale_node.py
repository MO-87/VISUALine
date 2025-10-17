import logging
from typing import Any, Dict

import cv2
import numpy as np

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class GrayscaleNode(NodeBase):
    """A node that converts a batch of frames to grayscale."""

    ## This node uses OpenCV/NumPy, not PyTorch.
    ## The TaskExecuter will pass it a NumPy array batch.
    use_torch: bool = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("GrayscaleNode initialized.")

    def process(self, batch: np.ndarray) -> np.ndarray:
        """
        Converts a batch of frames to grayscale.

        Args:
            batch (np.ndarray): The input batch as an RGB NumPy array
                                of shape (B, H, W, C).

        Returns:
            np.ndarray: The batch of grayscaled frames.
        """
        ## must iterate through the batch.. as cv2.cvtColor works on single images
        processed_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in batch]
        stacked_batch = np.stack(processed_frames)

        return np.expand_dims(stacked_batch, axis=-1)