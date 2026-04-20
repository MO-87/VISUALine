from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class NodeBase(ABC):
	"""
	Abstract Base Class for a pipeline processing node.
	
	Each node is responsible for its own setup, processing, and teardown.
	Nodes work exclusively with torch.Tensor data for efficient 
	end-to-end GPU pipeline execution.
	"""

	@property
	def fps_multiplier(self) -> float:
		"""
		Returns the factor by which this node changes the framerate.
		1.0 means no change. 2.0 means 2x FPS.
		"""
		return 1.0

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
	def process(self, data: torch.Tensor) -> torch.Tensor:
		"""
		Processes a batch of data through the node's operation.
		
		Input: torch.Tensor of shape (B, C, H, W), dtype float32
		Output: torch.Tensor of shape (B, C, H, W), dtype float32

		Args:
			data (torch.Tensor): The input batch of frames or images.

		Returns:
			torch.Tensor: The processed batch, in the same format it was received.
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

	def validate_input(self, data: torch.Tensor, expected_channels: int = 3) -> None:
		"""
		Helper method to validate input data shape and format.
		
		Raises appropriate exceptions if data doesn't match expected format.
		Can be called at the start of process() for debugging or validation.
		
		Args:
			data (torch.Tensor): Input data to validate.
			expected_channels (int): Expected number of channels (default: 3 for RGB).
		
		Raises:
			ValueError: If data shape or format is invalid.
			TypeError: If data type doesn't match torch.Tensor.
		"""
		if not isinstance(data, torch.Tensor):
			raise TypeError(
				f"{self.node_name} expects torch.Tensor input, "
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

	def __repr__(self) -> str:
		"""String representation of the node for logging and debugging."""
		return f"{self.node_name}(config={self.config})"