from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobResponse(BaseModel):
    """Returned immediately when a task is accepted."""
    job_id: str = Field(..., description="Unique identifier for the tracking the task")
    status: JobStatus = Field(default=JobStatus.PENDING)
    message: str = Field(..., description="Human readable status message")

class JobProgress(BaseModel):
    """Streamed via WebSockets or fetched via polling to show real-time progress."""
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, description="Percentage of completion (0.0 to 100.0)")
    current_frame: Optional[int] = Field(None, description="Current frame/image being processed")
    total_frames: Optional[int] = Field(None, description="Total number of frames/images to process")
    error_message: Optional[str] = Field(None, description="Error details if the job failed")