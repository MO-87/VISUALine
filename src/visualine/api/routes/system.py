import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from visualine.api.schemas import (
    SystemHealthResponse,
    JobProgress,
    JobStatus,
)
from visualine.api.services.system_service import (
    get_system_health,
    get_job_progress,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System & Jobs"])


TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
}

# If your JobStatus enum includes CANCELLED, support it without breaking older versions.
if hasattr(JobStatus, "CANCELLED"):
    TERMINAL_JOB_STATUSES.add(JobStatus.CANCELLED)


@router.get(
    "/api/v1/system/health",
    response_model=SystemHealthResponse,
)
async def system_health() -> SystemHealthResponse:
    """
    Return VISUALine backend/system health.

    Useful for:
    - UI dashboard
    - GPU/VRAM display
    - checking whether backend is online
    - showing active jobs / loaded runtime state
    """
    return get_system_health()


@router.get(
    "/api/v1/jobs/{job_id}/progress",
    response_model=JobProgress,
)
async def job_progress(job_id: str) -> JobProgress:
    """
    Polling endpoint for job progress.

    Useful as a fallback when WebSockets are unavailable.
    """
    progress = get_job_progress(job_id)

    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        )

    return progress


@router.websocket("/ws/progress/{job_id}")
async def job_progress_ws(
    websocket: WebSocket,
    job_id: str,
    interval: Optional[float] = Query(
        default=0.25,
        ge=0.05,
        le=5.0,
        description="Progress update interval in seconds.",
    ),
):
    """
    Real-time WebSocket endpoint for job progress.

    Frontend example:
        const ws = new WebSocket(`ws://localhost:8000/ws/progress/${jobId}`);
    """
    await websocket.accept()

    logger.info(f"WebSocket connected for job_id: {job_id}")

    try:
        while True:
            job_progress = get_job_progress(job_id)

            if job_progress is None:
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "status": "not_found",
                        "error": "Job not found",
                    }
                )
                break

            await websocket.send_json(
                job_progress.model_dump(mode="json")
            )

            if job_progress.status in TERMINAL_JOB_STATUSES:
                logger.info(
                    f"Job {job_id} reached terminal state "
                    f"({job_progress.status}). Closing WebSocket."
                )
                break

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client for job_id: {job_id}")

    except Exception as e:
        logger.error(
            f"WebSocket error for job {job_id}: {e}",
            exc_info=True,
        )

        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "status": "error",
                        "error": str(e),
                    }
                )
            except Exception:
                pass

    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass