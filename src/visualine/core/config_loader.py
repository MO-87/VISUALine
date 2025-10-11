from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoadingError(Exception):
    """Custom exception for all configuration loading errors."""
    pass


class BaseConfigLoader(ABC):
    """Abstract base class for all configuration loaders."""

    @abstractmethod
    def load(self, config_path: Path) -> Dict[str, Any]:
        """
        Loads a configuration file from the given path.

        Args:
            config_path (Path): The path to the configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary.

        Raises:
            ConfigLoadingError: If the file cannot be found or parsed.
        """
        pass


class YamlConfigLoader(BaseConfigLoader):
    """A concrete implementation for loading YAML configuration files."""

    def load(self, config_path: Path) -> Dict[str, Any]:
        """
        Loads and parses a YAML file, ensuring it's a valid dictionary.

        Args:
            config_path (Path): The path to the YAML file.

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary.

        Raises:
            ConfigLoadingError: If the file is not found, contains invalid YAML,
                                or the root element is not a dictionary.
        """
        if not config_path.is_file():
            raise ConfigLoadingError(f"Configuration file not found at: {config_path}")

        try:
            with config_path.open('r') as f:
                config_data = yaml.safe_load(f)
                if not isinstance(config_data, dict):
                    raise ConfigLoadingError(
                        f"Configuration content in '{config_path}' must be a dictionary (key-value pairs)."
                    )
                return config_data
        except yaml.YAMLError as e:
            raise ConfigLoadingError(f"Error parsing YAML file '{config_path}': {e}")
        except Exception as e:
            raise ConfigLoadingError(f"An unexpected error occurred while loading '{config_path}': {e}")