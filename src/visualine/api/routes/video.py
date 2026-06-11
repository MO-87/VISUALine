import logging

from fastapi import APIRouter, HTTPException, status

from visualine.api.schemas import JobResponse, VideoProcessRequest
from visualine.api.services.video_service import process_video

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/video",
    tags=["Video Processing"],
)


@router.post(
    "/process",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_video_endpoint(request: VideoProcessRequest) -> JobResponse:
    """
    Queue a video processing job.

    The backend returns a job ID immediately while video processing continues
    in the background through FFmpeg/CUDA/PipelineManager.
    """
    logger.info(
        "Received video processing request. "
        f"input={request.input_path}, "
        f"pipeline_id={request.pipeline_id}, "
        f"pipeline_config_path={request.pipeline_config_path}"
    )

    try:
        return await process_video(request)

    except FileNotFoundError as e:
        logger.warning(f"Video processing request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    except ValueError as e:
        logger.warning(f"Invalid video processing request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.error(
            f"Unexpected error while queuing video processing job: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue video processing job.",
        ) from e