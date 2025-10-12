from abc import ABC, abstractmethod
from typing import Any

class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers."""
    
    # This flag will be used by the TaskExecuter to identify PyTorch models.
    use_torch: bool = True

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Runs the model inference on the given data."""
        pass
        
    def cleanup(self):
        """Optional method for custom resource cleanup before eviction."""
        pass