import logging
from typing import Any, Dict

import cv2
import numpy as np

from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class GrayscaleNode(NodeBase):
    """A node that converts a batch of frames to grayscale."""

    ## This node uses OpenCV/NumPy, not PyTorch.
    ## The TaskExecuter will handle the data conversion automatically.
    use_torch: bool = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info("GrayscaleNode initialized.")

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts a single frame to grayscale.

        Args:
            frame (np.ndarray): The input frame as an RGB NumPy array.

        Returns:
            np.ndarray: The grayscaled frame.
        """
        # Note: The input from the manager is RGB, so we convert from RGB to GRAY.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)