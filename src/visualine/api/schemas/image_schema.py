from pydantic import BaseModel, Field

class ImageProcessRequest(BaseModel):
    """Request schema for processing a single image."""
    input_path: str = Field(
        ..., 
        description="Absolute OS path to the input image"
    )
    output_path: str = Field(
        ..., 
        description="Absolute OS path to save the processed image"
    )
    pipeline_config_path: str = Field(
        ..., 
        description="Absolute OS path to the YAML pipeline configuration file",
        json_schema_extra={"example": "C:/App/configs/pipeline_configs/test_upscale.yaml"}
    )

class ImageBatchProcessRequest(BaseModel):
    """Request schema for processing a directory of images."""
    input_dir: str = Field(
        ..., 
        description="Absolute OS path to the directory containing input images"
    )
    output_dir: str = Field(
        ..., 
        description="Absolute OS path to the directory where processed images will be saved"
    )
    pipeline_config_path: str = Field(
        ..., 
        description="Absolute OS path to the YAML pipeline configuration file",
        json_schema_extra={"example": "C:/App/configs/pipeline_configs/test_upscale.yaml"}
    )