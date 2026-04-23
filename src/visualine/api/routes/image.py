import logging
from fastapi import APIRouter, status

from visualine.api.schemas import ImageProcessRequest, ImageBatchProcessRequest, JobResponse
from visualine.api.services.image_service import process_single_image, process_image_batch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/image", tags=["Image Processing"])

@router.post("/process", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_image_endpoint(request: ImageProcessRequest):
    """
    Accepts a single image processing request..
    Returns a Job ID immediately while processing happens in the background
    """
    logger.info(f"Received single image processing request for: {request.input_path}")
    return await process_single_image(request)

@router.post("/process-batch", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_image_batch_endpoint(request: ImageBatchProcessRequest):
    """
    Accepts a directory of images for batch processing..
    Returns a Job ID immediately while processing happens in the background
    """
    logger.info(f"Received image batch processing request for directory: {request.input_dir}")
    return await process_image_batch(request)