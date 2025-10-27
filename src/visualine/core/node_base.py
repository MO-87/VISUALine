from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import torch
import numpy as np


class NodeBase(ABC):
    """
    Abstract Base Class for a pipeline processing node.
    
    Each node is responsible for its own setup, processing, and teardown.
    Nodes can work with either torch.Tensor or np.ndarray data, controlled
    by the `use_torch` flag for efficient pipeline execution.
    """
    
    ## Flag for the TaskExecuter to identify node type.
    ## Set to False in subclasses that only work with NumPy arrays.
    ## Set to True (default) for nodes that work with torch tensors.
    use_torch: bool = True

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the node with its specific configuration.
        
        Args:
            config (Dict[str, Any]): Node-specific configuration parameters.
        """
        self.config = config
        self.node_name = self.__class__.__name__
        self.is_setup = False

    def setup(self, device: torch.device) -> None:
        """
        One-time setup for the node. Called once before processing starts.
        
        Use this to load models, allocate resources, initialize buffers, etc.
        Base implementation sets is_setup flag; override and call super() if needed.

        Args:
            device (torch.device): The primary compute device (e.g., 'cuda' or 'cpu').
        """
        self.is_setup = True

    @abstractmethod
    def process(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Processes a batch of data through the node's operation.
        
        The type of `data` will be a torch.Tensor if use_torch is True,
        and a np.ndarray if use_torch is False. The TaskExecuter handles
        format conversions automatically based on the use_torch flag.
        
        For torch nodes:
            - Input: torch.Tensor of shape (B, C, H, W), dtype float32
            - Output: torch.Tensor of shape (B, C, H, W), dtype float32
        
        For numpy nodes:
            - Input: np.ndarray of shape (B, H, W, C), dtype uint8
            - Output: np.ndarray of shape (B, H, W, C), dtype uint8

        Args:
            data (Union[torch.Tensor, np.ndarray]): The input batch of frames or images.

        Returns:
            Union[torch.Tensor, np.ndarray]: The processed batch, in the same format it was received.
        """
        pass

    def teardown(self) -> None:
        """
        One-time cleanup for the node. Called once after processing is finished.
        
        Use this to release memory, clear CUDA cache, close file handles,
        or perform any other cleanup operations. Base implementation resets
        is_setup flag; override and call super() if needed.
        """
        self.is_setup = False

    def validate_input(self, data: Union[torch.Tensor, np.ndarray], 
                       expected_channels: int = 3) -> None:
        """
        Helper method to validate input data shape and format.
        
        Raises appropriate exceptions if data doesn't match expected format.
        Can be called at the start of process() for debugging or validation.
        
        Args:
            data (Union[torch.Tensor, np.ndarray]): Input data to validate.
            expected_channels (int): Expected number of channels (default: 3 for RGB).
        
        Raises:
            ValueError: If data shape or format is invalid.
            TypeError: If data type doesn't match use_torch flag.
        """
        if self.use_torch:
            if not isinstance(data, torch.Tensor):
                raise TypeError(
                    f"{self.node_name} expects torch.Tensor input (use_torch=True), "
                    f"but got {type(data).__name__}"
                )
            if data.ndim != 4:
                raise ValueError(
                    f"{self.node_name} expects 4D tensor (B, C, H, W), "
                    f"but got shape {data.shape}"
                )
            if data.shape[1] != expected_channels:
                raise ValueError(
                    f"{self.node_name} expects {expected_channels} channels, "
                    f"but got {data.shape[1]} channels"
                )
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"{self.node_name} expects np.ndarray input (use_torch=False), "
                    f"but got {type(data).__name__}"
                )
            if data.ndim != 4:
                raise ValueError(
                    f"{self.node_name} expects 4D array (B, H, W, C), "
                    f"but got shape {data.shape}"
                )
            if data.shape[3] != expected_channels:
                raise ValueError(
                    f"{self.node_name} expects {expected_channels} channels, "
                    f"but got {data.shape[3]} channels"
                )

    def __repr__(self) -> str:
        """String representation of the node for logging and debugging."""
        return f"{self.node_name}(config={self.config}, use_torch={self.use_torch})"