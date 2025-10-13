import logging
from pathlib import Path

# Important: We need to set up the logger before other imports
# to ensure all modules use the same configuration.
from visualine.core.config_loader import YamlConfigLoader
from visualine.core.logger import setup_logger

# Find the project root to correctly locate config and data files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# 1. Setup Logging
# We'll create a simple config here for the test. In a real app,
# this would be loaded from base_config.yaml.
log_config = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
setup_logger(log_config, LOGS_DIR)

# Get the logger instance for this specific file
logger = logging.getLogger(__name__)

# Now we can import the rest of our system
from visualine.core.pipeline_manager import PipelineManager


def main():
    """Main function to run the test pipeline."""
    logger.info("========== Starting VISUALine Pipeline Test ==========")

    # Define input and output paths
    input_video = DATA_DIR / "input" / "sample.mp4"
    output_video = DATA_DIR / "output" / "sample_grayscaled.mp4"
    pipeline_config = CONFIGS_DIR / "pipeline_configs" / "test_grayscale.yaml"

    # Check if input files exist
    if not input_video.exists():
        logger.critical(f"Input video not found! Please place 'sample.mp4' in '{input_video.parent}'")
        return
    if not pipeline_config.exists():
        logger.critical(f"Pipeline config not found at: {pipeline_config}")
        return

    # 2. Initialize Core Components
    config_loader = YamlConfigLoader()
    manager = PipelineManager(config_loader=config_loader)

    # 3. Load and Run the Pipeline
    try:
        manager.load_pipeline(pipeline_config_path=pipeline_config)
        manager.run(input_path=input_video, output_path=output_video)
    except Exception as e:
        logger.critical(f"An error occurred during the pipeline run: {e}", exc_info=True)
    
    logger.info("========== VISUALine Pipeline Test Finished ==========")


if __name__ == "__main__":
    main()