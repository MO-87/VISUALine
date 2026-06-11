import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

from visualine.api.schemas import (
    SystemHealthResponse,
    JobProgress,
    JobStatus,
    OptimizeStatusResponse,
)
from visualine.api.services.system_service import (
    get_system_health,
    get_job_progress,
)
from visualine.api.services.optimization_service import (
    get_optimized_status,
    run_optimization_generator,
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


from pathlib import Path
import os

from visualine.api.schemas import (
    SystemHealthResponse,
    JobProgress,
    JobStatus,
    OptimizeStatusResponse,
    BrowserExploreResponse,
    BrowserFileEntry,
)

# Project root for file browsing
PROJECT_ROOT = Path(__file__).resolve().parents[4]

@router.get("/api/v1/system/search")
async def search_file(name: str = Query(..., description="File name to search for")):
    """
    Search for a file by name recursively in data directories.
    """
    search_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT,
    ]
    
    for root in search_roots:
        if not root.exists(): continue
        # Use glob to find file recursively
        for match in root.rglob(name):
            if match.is_file():
                return {"path": str(match.resolve())}
            
    raise HTTPException(status_code=404, detail=f"File '{name}' not found in project data directories.")

@router.get(
    "/api/v1/system/explore",
    response_model=BrowserExploreResponse,
)
async def explore_fs(path: Optional[str] = Query(None)) -> BrowserExploreResponse:
    """
    Explore the local file system.
    """
    if path:
        # Support ~ for home
        if path.startswith("~"):
            target_path = Path.home() / path[2:] if len(path) > 1 else Path.home()
        else:
            target_path = Path(path)
        
        current_path = target_path.resolve()
    else:
        current_path = PROJECT_ROOT.resolve()

    if not current_path.exists() or not current_path.is_dir():
        # Fallback to home if path is invalid or inaccessible
        current_path = Path.home().resolve()

    parent_path = str(current_path.parent) if current_path.parent != current_path else None
    
    entries = []
    try:
        # We use scandir for performance
        with os.scandir(current_path) as it:
            for entry in it:
                try:
                    is_dir = entry.is_dir()
                    entries.append(BrowserFileEntry(
                        name=entry.name,
                        path=str(Path(entry.path).resolve()),
                        is_dir=is_dir,
                        size_bytes=entry.stat().st_size if not is_dir else None
                    ))
                except (PermissionError, OSError):
                    continue
    except (PermissionError, OSError) as e:
        raise HTTPException(status_code=403, detail=f"Cannot access directory: {e}")

    # Sort: directories first, then alphabetically
    entries.sort(key=lambda x: (not x.is_dir, x.name.lower()))

    return BrowserExploreResponse(
        current_path=str(current_path),
        parent_path=parent_path,
        entries=entries
    )


@router.get(
    "/api/v1/system/health",
    response_model=SystemHealthResponse,
)
async def system_health() -> SystemHealthResponse:
    """
    Return VISUALine backend/system health.
    """
    return get_system_health()


@router.get(
    "/api/v1/system/models",
    response_model=OptimizeStatusResponse,
)
async def system_models() -> OptimizeStatusResponse:
    """
    Check which models are optimized for the current hardware.
    """
    return await get_optimized_status()


@router.get("/api/v1/system/optimize/stream")
async def optimize_stream(
    model_type: str = Query(..., description="Model type to optimize"),
    tile_size: int = Query(64, description="Tile size"),
    padding: int = Query(16, description="Padding"),
    batch_size: int = Query(16, description="Batch size"),
):
    """
    Stream optimization logs in real-time using SSE.
    """
    return StreamingResponse(
        run_optimization_generator(model_type, tile_size, padding, batch_size),
        media_type="text/event-stream",
    )


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