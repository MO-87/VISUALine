import logging
from fastapi import APIRouter, status

from visualine.api.schemas import VideoProcessRequest, JobResponse
from visualine.api.services.video_service import process_video

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/video", tags=["Video Processing"])

@router.post("/process", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_video_endpoint(request: VideoProcessRequest):
    """
    Accepts a video processing request..
    Returns a Job ID immediately while video renders via FFmpeg & CUDA in the background
    """
    logger.info(f"Received video processing request for: {request.input_path}")
    return await process_video(request)