import logging
from pathlib import Path

logger = logging.getLogger(__name__)

## this should find the project's root directory (where pyproject.toml is)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = PROJECT_ROOT / "weights"
EXPORTS_DIR = PROJECT_ROOT / "exports"


def get_model_path(model_filename: str) -> Path:
    """
    Constructs the full path to a model weight file.

    Args:
        model_filename (str): The name of the model file -> "model_name.pth".

    Returns:
        Path: The full, absolute path to the model file.

    Raises:
        FileNotFoundError: If the model file does not exist in the weights or exports directory.
    """
    model_path_weights = WEIGHTS_DIR / model_filename
    model_path_exports = EXPORTS_DIR / model_filename

    if model_path_weights.is_file():
        model_path = model_path_weights
    elif model_path_exports.is_file():
        model_path = model_path_exports
    else:
        raise FileNotFoundError(
            f"Model file '{model_filename}' not found in:\n"
            f"  - {WEIGHTS_DIR}\n"
            f"  - {EXPORTS_DIR}"
        )
    
    logger.debug(f"Resolved model path: {model_path}")
    return model_path