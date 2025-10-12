import logging
from pathlib import Path

logger = logging.getLogger(__name__)

## this should find the project's root directory (where pyproject.toml is)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def get_model_path(model_filename: str) -> Path:
    """
    Constructs the full path to a model weight file.

    Args:
        model_filename (str): The name of the model file -> "model_name.pth".

    Returns:
        Path: The full, absolute path to the model file.

    Raises:
        FileNotFoundError: If the model file does not exist in the weights directory.
    """
    if not WEIGHTS_DIR.exists():
        raise FileNotFoundError(f"The global 'weights' directory does not exist at: {WEIGHTS_DIR}")

    model_path = WEIGHTS_DIR / model_filename
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.debug(f"Resolved model path: {model_path}")
    return model_path