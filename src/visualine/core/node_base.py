from abc import ABC, abstractmethod
from typing import Any, Dict

import torch  ## the NodeBase is now based on PyTorch Tensors

class NodeBase(ABC):
    """
    Abstract Base Class for a pipeline processing node.

    Each node in a pipeline must inherit from this class and implement the
    `process` method, which operates on a batch of data as a PyTorch Tensor.
    """
    
    ## Flag for the TaskExecuter to identify node type.
    ## Set to False in subclasses that only work with NumPy. (handled by TaskExecuter)
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the node with its specific configuration.
        """
        self.config = config
        self.node_name = self.__class__.__name__

    @abstractmethod
    def process(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of data.

        Args:
            batch (torch.Tensor): The input batch of frames or images as a
                                  PyTorch Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: The processed batch, also as a PyTorch Tensor.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.node_name}(config={self.config})"