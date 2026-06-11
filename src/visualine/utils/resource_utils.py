from pathlib import Path
from typing import Union

def get_resource_path(relative_path: Union[str, Path]) -> Path:
    """
    Simple resource path resolver.
    Returns the absolute path to a resource file.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    
    # In this project, resources are usually relative to the project root
    # We'll assume the caller passes a path that can be found from CWD or project root
    return path.absolute()
