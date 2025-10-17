from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import numpy as np

class NodeBase(ABC):
    """
    Abstract Base Class for a pipeline processing node.
    Each node is responsible for its own setup, processing, and teardown.
    """
    
    ## Flag for the TaskExecuter to identify node type.
    ## Set to False in subclasses that only work with NumPy arrays.
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the node with its specific configuration.
        """
        self.config = config
        self.node_name = self.__class__.__name__
        self.is_setup = False

    def setup(self, device: torch.device):
        """
        One-time setup for the node. Called once before processing starts.
        Use this to load models, allocate resources, etc.

        Args:
            device (torch.device): The primary compute device (e.g., 'cuda' or 'cpu').
        """
        self.is_setup = True
        pass

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Processes a batch of data. The type of `data` will be a torch.Tensor
        if use_torch is True, and a np.ndarray if use_torch is False.

        Args:
            data (Any): The input batch of frames or images.

        Returns:
            Any: The processed batch, in the same format it was received.
        """
        pass

    def teardown(self):
        """
        One-time cleanup for the node. Called once after processing is finished.
        Use this to release memory or other resources.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.node_name}(config={self.config})"