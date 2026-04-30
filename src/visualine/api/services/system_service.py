import logging
import uuid
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional

import torch

from visualine.api.schemas import (
    HardwareInfo,
    JobProgress,
    JobRuntimeInfo,
    JobStatus,
    JobStatusResponse,
    JobType,
    PipelineRuntimeInfo,
    SystemHealthResponse,
)
from visualine.core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


# In-memory job registry.
# Good enough for a local desktop / Electron / Streamlit-backed app.
ACTIVE_JOBS: Dict[str, JobStatusResponse] = {}
_JOBS_LOCK = Lock()

_APP_STARTED_AT = datetime.utcnow()


def create_job(
    pipeline_id: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    job_type: Optional[JobType] = None,
    input_filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a tracking ID and initialize the job state.
    """
    job_id = str(uuid.uuid4())

    now = datetime.utcnow()

    job = JobStatusResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0.0,
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        job_type=job_type,
        input_filename=input_filename,
        message="Job queued.",
        created_at=now,
        metadata=metadata or {},
    )

    with _JOBS_LOCK:
        ACTIVE_JOBS[job_id] = job

    logger.info(f"Created job {job_id} for pipeline={pipeline_id}")
    return job_id


def get_job_status(job_id: str) -> Optional[JobStatusResponse]:
    """
    Return the full job state.
    """
    with _JOBS_LOCK:
        return ACTIVE_JOBS.get(job_id)


def get_job_progress(job_id: str) -> Optional[JobProgress]:
    """
    Return a lightweight progress object for polling/WebSocket updates.
    """
    with _JOBS_LOCK:
        job = ACTIVE_JOBS.get(job_id)

    if job is None:
        return None

    return JobProgress(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        current_frame=job.current_frame,
        total_frames=job.total_frames,
        message=job.message,
        error_message=job.error_message,
    )


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    progress: Optional[float] = None,
    current_frame: Optional[int] = None,
    total_frames: Optional[int] = None,
    message: Optional[str] = None,
    error_message: Optional[str] = None,
    output_filename: Optional[str] = None,
    output_url: Optional[str] = None,
    preview_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[JobStatusResponse]:
    """
    Update a tracked job.

    This is intentionally flexible so video/image services can reuse it.
    """
    with _JOBS_LOCK:
        job = ACTIVE_JOBS.get(job_id)

        if job is None:
            logger.warning(f"Tried to update unknown job: {job_id}")
            return None

        old_status = job.status

        if status is not None:
            job.status = status

        if progress is not None:
            job.progress = max(0.0, min(100.0, float(progress)))

        if current_frame is not None:
            job.current_frame = max(0, int(current_frame))

        if total_frames is not None:
            job.total_frames = max(0, int(total_frames))

        if message is not None:
            job.message = message

        if error_message is not None:
            job.error_message = error_message

        if output_filename is not None:
            job.output_filename = output_filename

        if output_url is not None:
            job.output_url = output_url

        if preview_url is not None:
            job.preview_url = preview_url

        if metadata:
            job.metadata.update(metadata)

        now = datetime.utcnow()

        if old_status != JobStatus.PROCESSING and job.status == JobStatus.PROCESSING:
            job.started_at = now

        if job.status in _terminal_statuses():
            job.completed_at = now

            if job.started_at is not None:
                job.runtime_seconds = (job.completed_at - job.started_at).total_seconds()

        return job


def mark_job_processing(job_id: str, message: str = "Processing...") -> None:
    update_job(
        job_id,
        status=JobStatus.PROCESSING,
        progress=0.0,
        message=message,
    )


def mark_job_completed(
    job_id: str,
    *,
    output_filename: Optional[str] = None,
    output_url: Optional[str] = None,
    preview_url: Optional[str] = None,
    message: str = "Processing completed.",
) -> None:
    update_job(
        job_id,
        status=JobStatus.COMPLETED,
        progress=100.0,
        output_filename=output_filename,
        output_url=output_url,
        preview_url=preview_url,
        message=message,
    )


def mark_job_failed(job_id: str, error_message: str) -> None:
    update_job(
        job_id,
        status=JobStatus.FAILED,
        message="Processing failed.",
        error_message=error_message,
    )


def mark_job_cancelled(job_id: str, message: str = "Job cancelled.") -> None:
    if hasattr(JobStatus, "CANCELLED"):
        update_job(
            job_id,
            status=JobStatus.CANCELLED,
            message=message,
        )
    else:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            message=message,
            error_message="Job was cancelled.",
        )


def make_progress_callback(job_id: str):
    """
    Create a callback compatible with PipelineManager.run(..., progress_callback=...).

    PipelineManager may call:
        callback(current, total)
    or:
        callback(current, total, status)
    """

    def _callback(current: int, total: int, status: Optional[str] = None) -> None:
        if total and total > 0:
            progress = (current / total) * 100.0
        else:
            progress = 0.0

        update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            current_frame=current,
            total_frames=total,
            message=status or "Processing...",
        )

    return _callback


def delete_job(job_id: str) -> bool:
    """
    Remove a job from the in-memory registry.
    Useful for cleanup after the UI is done with a result.
    """
    with _JOBS_LOCK:
        existed = job_id in ACTIVE_JOBS
        ACTIVE_JOBS.pop(job_id, None)

    return existed


def get_system_health() -> SystemHealthResponse:
    """
    Fetch real-time hardware/runtime status.

    Safe enough for frequent UI polling.
    """
    rm = ResourceManager()

    hardware = _get_hardware_info(rm)
    jobs = _get_job_runtime_info()
    runtime = _get_pipeline_runtime_info(rm)

    uptime = (datetime.utcnow() - _APP_STARTED_AT).total_seconds()

    return SystemHealthResponse(
        status="online",
        app_name="VISUALine",
        version=None,
        uptime_seconds=uptime,
        hardware=hardware,
        runtime=runtime,
        jobs=jobs,
    )


def _get_hardware_info(rm: ResourceManager) -> HardwareInfo:
    device = "cpu"
    device_name = None
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    torch_version = torch.__version__

    vram_total = None
    vram_allocated = None
    vram_reserved = None
    vram_limit = None
    vram_usage_percent = None

    if cuda_available:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)

        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024**3)
        vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)

        vram_limit_raw = getattr(rm, "vram_limit_gb", None)
        if vram_limit_raw is not None and vram_limit_raw != float("inf"):
            vram_limit = float(vram_limit_raw)

        if vram_total and vram_total > 0:
            vram_usage_percent = (vram_allocated / vram_total) * 100.0

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon"

    return HardwareInfo(
        device=device,
        device_name=device_name,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        torch_version=torch_version,
        vram_total_gb=vram_total,
        vram_allocated_gb=vram_allocated,
        vram_reserved_gb=vram_reserved,
        vram_limit_gb=vram_limit,
        vram_usage_percent=vram_usage_percent,
        is_locked=bool(getattr(rm, "_is_locked", False)),
    )


def _get_job_runtime_info() -> JobRuntimeInfo:
    with _JOBS_LOCK:
        jobs = list(ACTIVE_JOBS.values())

    active_statuses = {JobStatus.PENDING, JobStatus.PROCESSING}

    if hasattr(JobStatus, "QUEUED"):
        queued_status = JobStatus.QUEUED
    else:
        queued_status = JobStatus.PENDING

    return JobRuntimeInfo(
        active_jobs_count=sum(1 for job in jobs if job.status in active_statuses),
        queued_jobs_count=sum(1 for job in jobs if job.status == queued_status),
        completed_jobs_count=sum(1 for job in jobs if job.status == JobStatus.COMPLETED),
        failed_jobs_count=sum(1 for job in jobs if job.status == JobStatus.FAILED),
    )


def _get_pipeline_runtime_info(rm: ResourceManager) -> PipelineRuntimeInfo:
    """
    Best-effort model cache introspection.

    This avoids depending too tightly on ResourceManager internals.
    """
    cached_models = []

    for attr_name in ["_model_cache", "_models", "models", "cache"]:
        maybe_cache = getattr(rm, attr_name, None)

        if isinstance(maybe_cache, dict):
            cached_models = list(maybe_cache.keys())
            break

    return PipelineRuntimeInfo(
        is_pipeline_loaded=False,
        loaded_pipeline_path=None,
        loaded_pipeline_name=None,
        loaded_nodes=[],
        cached_models_count=len(cached_models),
        cached_models=[str(name) for name in cached_models],
    )


def _terminal_statuses() -> set:
    statuses = {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
    }

    if hasattr(JobStatus, "CANCELLED"):
        statuses.add(JobStatus.CANCELLED)

    return statuses