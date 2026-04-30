from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    BATCH = "batch"


class JobResponse(BaseModel):
    """
    Returned immediately when a processing task is accepted.
    """

    job_id: str = Field(..., description="Unique identifier for tracking the job")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    message: str = Field(..., description="Human-readable status message")

    pipeline_id: Optional[str] = Field(
        None,
        description="Pipeline/config identifier used for this job",
    )

    job_type: Optional[JobType] = Field(
        None,
        description="Type of processing job",
    )


class JobProgress(BaseModel):
    """
    Returned by polling or streamed through WebSockets/SSE.
    """

    job_id: str
    status: JobStatus

    progress: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of completion from 0.0 to 100.0",
    )

    current_frame: Optional[int] = Field(
        None,
        ge=0,
        description="Current frame/image being processed",
    )

    total_frames: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of frames/images to process",
    )

    message: Optional[str] = Field(
        None,
        description="Current human-readable progress message",
    )

    error_message: Optional[str] = Field(
        None,
        description="Error details if the job failed",
    )


class JobStatusResponse(BaseModel):
    """
    Full job state returned by GET /jobs/{job_id}.
    """

    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=100.0)

    pipeline_id: Optional[str] = None
    pipeline_name: Optional[str] = None
    job_type: Optional[JobType] = None

    input_filename: Optional[str] = None
    output_filename: Optional[str] = None

    output_url: Optional[str] = Field(
        None,
        description="URL used by the UI to preview or download the processed output",
    )

    preview_url: Optional[str] = Field(
        None,
        description="Optional lightweight preview URL",
    )

    current_frame: Optional[int] = Field(None, ge=0)
    total_frames: Optional[int] = Field(None, ge=0)

    message: Optional[str] = None
    error_message: Optional[str] = None

    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    runtime_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total processing time in seconds when available",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra job metadata for UI/debugging",
    )


class JobCreateRequest(BaseModel):
    """
    Metadata part of a job creation request.

    File upload itself should usually be handled via multipart/form-data
    in the route, while this schema describes the non-file options.
    """

    pipeline_id: str = Field(
        ...,
        description="Pipeline/config ID to run",
    )

    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime parameter overrides, usually generated from UI controls",
    )

    keep_original_audio: bool = Field(
        True,
        description="Whether to mux original audio back into processed videos",
    )


class JobCancelResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str