from typing import Optional
from pydantic import BaseModel, Field

class HardwareInfo(BaseModel):
    device: str = Field(..., description="Primary compute device (e.g., 'cuda', 'cpu', 'mps')")
    device_name: Optional[str] = Field(None, description="Name of the GPU if available")
    vram_total_gb: Optional[float] = Field(None, description="Total available VRAM in GB")
    vram_used_gb: Optional[float] = Field(None, description="Currently allocated VRAM in GB")
    vram_limit_gb: Optional[float] = Field(None, description="Soft limit for VRAM defined in ResourceManager")
    is_locked: bool = Field(..., description="Whether the ResourceManager is currently locked for execution")

class SystemHealthResponse(BaseModel):
    status: str = Field(default="online")
    hardware: HardwareInfo
    active_jobs_count: int = Field(default=0, description="Number of currently running processing jobs")