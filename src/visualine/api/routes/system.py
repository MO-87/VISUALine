import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from visualine.api.schemas import SystemHealthResponse, JobStatus
from visualine.api.services.system_service import get_system_health, get_job_progress

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System & Jobs"])

@router.get("/api/v1/system/health", response_model=SystemHealthResponse)
async def system_health():
    """Returns hardware usage (VRAM) and system status. Useful for an Electron dashboard."""
    return get_system_health()

@router.websocket("/ws/progress/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """
    Real-time WebSocket endpoint..
    The Electron frontend connects here using: new WebSocket(`ws://localhost:8000/ws/progress/${jobId}`)
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for job_id: {job_id}")

    try:
        while True:
            job_progress = get_job_progress(job_id)
            
            if not job_progress:
                await websocket.send_json({"error": "Job not found"})
                break
                
            await websocket.send_json(job_progress.model_dump())

            if job_progress.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                logger.info(f"Job {job_id} reached terminal state ({job_progress.status}). Closing WS.")
                break

            ## sleep briefly to avoid pegging the CPU (10 updates per second is smooth enough for UI)
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client for job_id: {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass ## socket was already closed