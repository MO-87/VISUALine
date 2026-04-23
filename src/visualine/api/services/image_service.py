import asyncio
import logging
import traceback
from pathlib import Path

from visualine.api.schemas import (
    ImageProcessRequest, ImageBatchProcessRequest, JobResponse, JobStatus
)
from visualine.api.services.system_service import create_job, ACTIVE_JOBS
from visualine.core.pipeline_manager import PipelineManager
from visualine.core.config_loader import YamlConfigLoader

logger = logging.getLogger(__name__)

def _execute_pipeline_sync(job_id: str, input_path: Path, output_path: Path, config_path: Path):
    """The synchronous blocking function that runs in the background thread."""
    try:
        ACTIVE_JOBS[job_id].status = JobStatus.PROCESSING
        
        def progress_callback(current: int, total: int):
            job = ACTIVE_JOBS[job_id]
            job.current_frame = current
            job.total_frames = total
            job.progress = (current / total * 100.0) if total > 0 else 0.0

        config_loader = YamlConfigLoader()
        pipeline = PipelineManager(config_loader=config_loader)
        
        pipeline.load_pipeline(config_path)
        pipeline.run(input_path, output_path, progress_callback=progress_callback)
        
        ACTIVE_JOBS[job_id].status = JobStatus.COMPLETED
        ACTIVE_JOBS[job_id].progress = 100.0
        
    except Exception as e:
        logger.error(f"Image pipeline job {job_id} failed: {e}")
        ACTIVE_JOBS[job_id].status = JobStatus.FAILED
        ACTIVE_JOBS[job_id].error_message = f"{str(e)}\n{traceback.format_exc()}"

async def process_single_image(request: ImageProcessRequest) -> JobResponse:
    job_id = create_job()
    
    asyncio.create_task(
        asyncio.to_thread(
            _execute_pipeline_sync, 
            job_id, 
            Path(request.input_path), 
            Path(request.output_path), 
            Path(request.pipeline_config_path)
        )
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Image processing task initiated."
    )

async def process_image_batch(request: ImageBatchProcessRequest) -> JobResponse:
    job_id = create_job()
    
    asyncio.create_task(
        asyncio.to_thread(
            _execute_pipeline_sync, 
            job_id, 
            Path(request.input_dir), 
            Path(request.output_dir), 
            Path(request.pipeline_config_path)
        )
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Image batch processing task initiated."
    )