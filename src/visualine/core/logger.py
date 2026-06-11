import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict


def setup_logger(log_config: Dict[str, Any], log_dir: Path) -> None:
    """
    Configures the application's logger based on the provided configuration.

    This function is designed to be called once at application startup.

    Args:
        log_config (Dict[str, Any]): A dictionary containing logging settings.
        log_dir (Path): The directory where log files will be stored.
    """
    # Get logger settings from the config, with sensible defaults
    log_level = log_config.get("level", "INFO").upper()
    log_format = log_config.get(
        "format", "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    )
    log_file = log_dir / log_config.get("filename", "visualine.log")
    max_bytes = log_config.get("max_bytes", 5 * 1024 * 1024)  ## 5 MB
    backup_count = log_config.get("backup_count", 5)

    # Ensure the log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get the logger for the entire 'visualine' package
    logger = logging.getLogger("visualine")
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate logs in parent loggers

    # Create a formatter
    formatter = logging.Formatter(log_format)

    # Don't add handlers if they already exist
    if not logger.handlers:
        # 1. Console Handler - to see logs in your terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Rotating File Handler - to save logs to a file
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.propagate = False
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    logger.info("Logger has been configured.")