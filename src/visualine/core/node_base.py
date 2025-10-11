from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class NodeBase(ABC):
    """
    Abstract Base Class for a pipeline processing node.

    Each node in a pipeline must inherit from this class and implement the
    `process` method. The __init__ method should be used to accept any
    node-specific configurations. This adheres to the Liskov Substitution Principle.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the node with its specific configuration. This follows
        the Dependency Inversion Principle, as the node receives its
        dependencies (config) upon creation.

        Args:
            config (Dict[str, Any]): A dictionary of configuration parameters
                                     for this node, loaded from a pipeline config.
        """
        self.config = config
        self.node_name = self.__class__.__name__

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray: ## choosing numpy specifically provides a universal choise..
        """
        Processes a single video frame. This method defines the core,
        non-negotiable action of any node.

        Args:
            frame (np.ndarray): The input video frame (as a NumPy array).

        Returns:
            np.ndarray: The processed video frame (as a NumPy array).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.node_name}(config={self.config})"