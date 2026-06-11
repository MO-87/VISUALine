from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HardwareInfo(BaseModel):
    device: str = Field(
        ...,
        description="Primary compute device, e.g. 'cuda', 'cpu', or 'mps'",
    )

    device_name: Optional[str] = Field(
        None,
        description="Name of the GPU/accelerator if available",
    )

    cuda_available: bool = Field(
        False,
        description="Whether CUDA is available",
    )

    cuda_version: Optional[str] = Field(
        None,
        description="CUDA version reported by PyTorch",
    )

    torch_version: Optional[str] = Field(
        None,
        description="Installed PyTorch version",
    )

    vram_total_gb: Optional[float] = Field(
        None,
        description="Total GPU VRAM in GB",
    )

    vram_allocated_gb: Optional[float] = Field(
        None,
        description="VRAM currently allocated by PyTorch in GB",
    )

    vram_reserved_gb: Optional[float] = Field(
        None,
        description="VRAM currently reserved/cached by PyTorch in GB",
    )

    vram_limit_gb: Optional[float] = Field(
        None,
        description="Soft VRAM limit defined by ResourceManager",
    )

    vram_usage_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Approximate VRAM usage percentage",
    )

    is_locked: bool = Field(
        False,
        description="Whether ResourceManager is currently locked for loading/execution",
    )


class PipelineRuntimeInfo(BaseModel):
    is_pipeline_loaded: bool = Field(
        False,
        description="Whether a pipeline is currently loaded in memory",
    )

    loaded_pipeline_path: Optional[str] = Field(
        None,
        description="Path or identifier of the currently loaded pipeline config",
    )

    loaded_pipeline_name: Optional[str] = Field(
        None,
        description="Human-readable name of the loaded pipeline",
    )

    loaded_nodes: List[str] = Field(
        default_factory=list,
        description="Names/classes of currently loaded nodes",
    )

    cached_models_count: int = Field(
        0,
        description="Number of models currently cached by ResourceManager",
    )

    cached_models: List[str] = Field(
        default_factory=list,
        description="Names/keys of cached models",
    )


class JobRuntimeInfo(BaseModel):
    active_jobs_count: int = Field(
        0,
        ge=0,
        description="Number of currently running processing jobs",
    )

    queued_jobs_count: int = Field(
        0,
        ge=0,
        description="Number of jobs waiting to be processed",
    )

    completed_jobs_count: int = Field(
        0,
        ge=0,
        description="Number of completed jobs currently tracked",
    )

    failed_jobs_count: int = Field(
        0,
        ge=0,
        description="Number of failed jobs currently tracked",
    )


class SystemHealthResponse(BaseModel):
    status: str = Field(
        default="online",
        description="Overall backend status",
    )

    app_name: str = Field(
        default="VISUALine",
        description="Application name",
    )

    version: Optional[str] = Field(
        None,
        description="Application version if available",
    )

    uptime_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Backend uptime in seconds",
    )

    hardware: HardwareInfo

    runtime: PipelineRuntimeInfo = Field(
        default_factory=PipelineRuntimeInfo,
        description="Current pipeline/model runtime state",
    )

    jobs: JobRuntimeInfo = Field(
        default_factory=JobRuntimeInfo,
        description="Current job queue/runtime state",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Time when this health response was generated",
    )


class SimpleMessageResponse(BaseModel):
    status: str = Field(default="ok")
    message: str


class OptimizeRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to optimize, e.g. 'span', 'realesr-anime'")
    tile_size: Optional[int] = Field(64, description="Tile size for optimization")
    padding: Optional[int] = Field(16, description="Padding per tile")
    batch_size: Optional[int] = Field(16, description="Optimal batch size for compilation")


class BrowserFileEntry(BaseModel):
    name: str = Field(..., description="File or directory name")
    path: str = Field(..., description="Absolute path on the local system")
    is_dir: bool = Field(..., description="Whether the entry is a directory")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")


class BrowserExploreResponse(BaseModel):
    current_path: str = Field(..., description="Current directory absolute path")
    parent_path: Optional[str] = Field(None, description="Parent directory absolute path")
    entries: List[BrowserFileEntry] = Field(default_factory=list, description="List of files and directories")


class OptimizeStatusResponse(BaseModel):
    span: bool = Field(False, description="Whether SPAN model is optimized")
    realesrgan: bool = Field(False, description="Whether RealESRGAN model is optimized")
    rife: bool = Field(False, description="Whether RIFE model is optimized")