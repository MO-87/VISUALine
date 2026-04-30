import logging
import mimetypes
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from visualine.api.schemas import JobStatus, JobStatusResponse
from visualine.api.services.system_service import delete_job, get_job_status

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["Jobs"],
)


def _get_job_or_404(job_id: str) -> JobStatusResponse:
    job = get_job_status(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return job


def _resolve_output_path(job: JobStatusResponse) -> Optional[Path]:
    """
    Resolve the final output path for a job.

    The services currently store paths in job.metadata under keys like:
    - actual_output_path
    - output_path
    - output_dir
    - requested_output_path

    We prefer actual_output_path because PipelineManager/FFmpeg may change suffix.
    """
    metadata = job.metadata or {}

    candidate_keys = [
        "actual_output_path",
        "output_path",
        "output_dir",
        "requested_output_path",
    ]

    for key in candidate_keys:
        value = metadata.get(key)

        if not value:
            continue

        path = Path(value)

        if path.exists:
            return path

    return None


def _make_zip_for_directory(directory: Path, job_id: str) -> Path:
    """
    Create a zip archive for batch outputs.
    """
    zip_root = Path(tempfile.gettempdir()) / "visualine_job_downloads"
    zip_root.mkdir(parents=True, exist_ok=True)

    zip_base = zip_root / f"{job_id}_outputs"
    zip_path = zip_base.with_suffix(".zip")

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        base_name=str(zip_base),
        format="zip",
        root_dir=str(directory),
    )

    return zip_path


def _delete_output_files(job: JobStatusResponse) -> None:
    output_path = _resolve_output_path(job)

    if output_path is None or not output_path.exists():
        return

    if output_path.is_dir():
        shutil.rmtree(output_path, ignore_errors=True)
    else:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
)
async def get_job_endpoint(job_id: str) -> JobStatusResponse:
    """
    Get full job status.

    UI uses this for:
    - status badge
    - progress bar
    - output URL
    - error messages
    - runtime metadata
    """
    return _get_job_or_404(job_id)


@router.get("/{job_id}/output")
async def get_job_output_endpoint(job_id: str):
    """
    Download or preview the completed job output.

    For single image/video jobs:
        returns the output file.

    For batch jobs:
        returns a zip archive of the output directory.
    """
    job = _get_job_or_404(job_id)

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=job.error_message or "Job failed.",
        )

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed yet. Current status: {job.status}",
        )

    output_path = _resolve_output_path(job)

    if output_path is None or not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file was not found for this job.",
        )

    if output_path.is_dir():
        zip_path = _make_zip_for_directory(output_path, job_id)

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_path.name,
        )

    media_type, _ = mimetypes.guess_type(str(output_path))

    return FileResponse(
        path=str(output_path),
        media_type=media_type or "application/octet-stream",
        filename=output_path.name,
    )


@router.delete("/{job_id}")
async def delete_job_endpoint(
    job_id: str,
    delete_files: bool = Query(
        default=False,
        description="If true, also delete the job output files/directories.",
    ),
):
    """
    Delete a job from the in-memory registry.

    This does not stop an already-running thread yet. It is mainly for UI cleanup
    after a job has completed or failed.
    """
    job = _get_job_or_404(job_id)

    if delete_files:
        try:
            _delete_output_files(job)
        except Exception as e:
            logger.warning(
                f"Failed to delete output files for job {job_id}: {e}",
                exc_info=True,
            )

    deleted = delete_job(job_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return {
        "status": "ok",
        "message": f"Job {job_id} deleted.",
    }