import uuid
import logging
from typing import Dict, Optional

import torch
from visualine.api.schemas import JobProgress, JobStatus, SystemHealthResponse, HardwareInfo
from visualine.core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

## GLOBAL in-memory state for tracking jobs.. (no need for Redis/Celery).
ACTIVE_JOBS: Dict[str, JobProgress] = {}

def create_job() -> str:
    """Generates a tracking ID and initializes the job state."""
    job_id = str(uuid.uuid4())
    ACTIVE_JOBS[job_id] = JobProgress(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0
    )
    return job_id

def get_job_progress(job_id: str) -> Optional[JobProgress]:
    """Retrieves the current state of a specific job."""
    return ACTIVE_JOBS.get(job_id)

def get_system_health() -> SystemHealthResponse:
    """Fetches real-time hardware utilization via the ResourceManager."""
    rm = ResourceManager()
    
    device = "cpu"
    device_name = None
    vram_total = 0.0
    vram_used = 0.0
    
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_used = rm._get_current_vram_gb()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon"

    hw_info = HardwareInfo(
        device=device,
        device_name=device_name,
        vram_total_gb=vram_total,
        vram_used_gb=vram_used,
        vram_limit_gb=rm.vram_limit_gb if rm.vram_limit_gb != float('inf') else None,
        is_locked=getattr(rm, '_is_locked', False)
    )
    
    ## counting active background threads..
    active_count = sum(1 for job in ACTIVE_JOBS.values() if job.status in [JobStatus.PENDING, JobStatus.PROCESSING])
    
    return SystemHealthResponse(
        status="online",
        hardware=hw_info,
        active_jobs_count=active_count
    )