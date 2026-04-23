from pydantic import BaseModel, Field

class VideoProcessRequest(BaseModel):
    input_path: str = Field(
        ..., 
        description="Absolute OS path to the input video file"
    )
    output_path: str = Field(
        ..., 
        description="Absolute OS path to save the processed video file"
    )
    pipeline_config_path: str = Field(
        ..., 
        description="Absolute OS path to the YAML pipeline configuration file",
        json_schema_extra={"example": "C:/App/configs/pipeline_configs/test_colorization.yaml"}
    )