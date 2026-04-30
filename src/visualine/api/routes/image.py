import logging

from fastapi import APIRouter, HTTPException, status

from visualine.api.schemas import (
    ImageBatchProcessRequest,
    ImageProcessRequest,
    JobResponse,
)
from visualine.api.services.image_service import (
    process_image_batch,
    process_single_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/image",
    tags=["Image Processing"],
)


@router.post(
    "/process",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_image_endpoint(request: ImageProcessRequest) -> JobResponse:
    """
    Queue a single-image processing job.

    The backend returns a job ID immediately while processing continues
    in the background.
    """
    logger.info(
        "Received single image processing request. "
        f"input={request.input_path}, "
        f"pipeline_id={request.pipeline_id}, "
        f"pipeline_config_path={request.pipeline_config_path}"
    )

    try:
        return await process_single_image(request)

    except FileNotFoundError as e:
        logger.warning(f"Image processing request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    except ValueError as e:
        logger.warning(f"Invalid image processing request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.error(
            f"Unexpected error while queuing image processing job: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue image processing job.",
        ) from e


@router.post(
    "/process-batch",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_image_batch_endpoint(
    request: ImageBatchProcessRequest,
) -> JobResponse:
    """
    Queue an image-directory batch processing job.

    The backend returns a job ID immediately while processing continues
    in the background.
    """
    logger.info(
        "Received image batch processing request. "
        f"input_dir={request.input_dir}, "
        f"pipeline_id={request.pipeline_id}, "
        f"pipeline_config_path={request.pipeline_config_path}"
    )

    try:
        return await process_image_batch(request)

    except FileNotFoundError as e:
        logger.warning(f"Image batch processing request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    except ValueError as e:
        logger.warning(f"Invalid image batch processing request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.error(
            f"Unexpected error while queuing image batch job: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue image batch processing job.",
        ) from e