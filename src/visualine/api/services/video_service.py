import asyncio
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from visualine.api.schemas import (
    JobResponse,
    JobStatus,
    JobType,
    VideoOutputFormat,
    VideoProcessRequest,
)
from visualine.api.services.system_service import (
    create_job,
    make_progress_callback,
    mark_job_completed,
    mark_job_failed,
    mark_job_processing,
    update_job,
)
from visualine.api.services.pipeline_service import resolve_pipeline_config_path
from visualine.core.config_loader import YamlConfigLoader
from visualine.core.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs" / "api"


def _queued_status() -> JobStatus:
    return JobStatus.QUEUED if hasattr(JobStatus, "QUEUED") else JobStatus.PENDING


def _resolve_pipeline_config_path(request: VideoProcessRequest) -> Path:
    """
    Resolve either:
    - pipeline_config_path: direct YAML path
    - pipeline_id: UI-friendly pipeline id/name
    """
    pipeline_ref = request.get_pipeline_reference()
    return resolve_pipeline_config_path(pipeline_ref)


def _generate_video_output_path(
    input_path: Path,
    job_id: str,
    output_format: VideoOutputFormat,
) -> Path:
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if output_format == VideoOutputFormat.SAME_AS_INPUT:
        suffix = input_path.suffix.lower()

        # GIF is processed as video, but output should be browser-friendly video.
        if suffix == ".gif":
            suffix = ".mp4"
    else:
        suffix = f".{output_format.value}"

    return job_dir / f"{input_path.stem}_processed{suffix}"


def _resolve_existing_output_path(output_path: Path) -> Path:
    """
    PipelineManager/VideoProcessor may change output suffix.

    Example:
        requested: output.mp4
        actual:    output.mov
    """
    output_path = Path(output_path)

    if output_path.exists():
        return output_path

    for suffix in [".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif"]:
        candidate = output_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    return output_path


def _materialize_runtime_config(
    base_config_path: Path,
    overrides: Dict[str, Any],
    job_id: str,
) -> Path:
    """
    Create a temporary YAML config with runtime UI overrides applied.

    Supports:
        {"prompt": "person", "blur_intensity": 41}

    Also supports explicit node override syntax:
        {"node.0.prompt": "person"}
        {"nodes.2.blur_intensity": 51}

    Simple overrides are applied to every node param with the same key.
    """
    if not overrides:
        return base_config_path

    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid pipeline config: {base_config_path}")

    pipeline = config.get("pipeline", [])

    if not isinstance(pipeline, list):
        raise ValueError(f"Pipeline config has invalid 'pipeline' field: {base_config_path}")

    for key, value in overrides.items():
        applied = False

        if key.startswith("node.") or key.startswith("nodes."):
            parts = key.split(".")

            if len(parts) == 3 and parts[1].isdigit():
                node_idx = int(parts[1])
                param_name = parts[2]

                if 0 <= node_idx < len(pipeline):
                    params = pipeline[node_idx].setdefault("params", {})
                    params[param_name] = value
                    applied = True

        if applied:
            continue

        for node_cfg in pipeline:
            params = node_cfg.setdefault("params", {})

            if key in params:
                params[key] = value
                applied = True

        if not applied:
            logger.warning(
                f"Override '{key}' did not match any existing node param in "
                f"{base_config_path.name}."
            )

    temp_dir = Path(tempfile.gettempdir()) / "visualine_runtime_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = temp_dir / f"{job_id}_{base_config_path.name}"

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return runtime_config_path


def _execute_video_pipeline_sync(
    job_id: str,
    input_path: Path,
    output_path: Path,
    config_path: Path,
) -> None:
    """
    Blocking worker function. Runs inside a background thread.
    """
    pipeline: Optional[PipelineManager] = None

    try:
        mark_job_processing(job_id, "Loading video pipeline...")

        config_loader = YamlConfigLoader()
        pipeline = PipelineManager(config_loader=config_loader)

        pipeline.load_pipeline(config_path)

        progress_callback = make_progress_callback(job_id)

        update_job(
            job_id,
            message="Processing video...",
            metadata={
                "input_path": str(input_path),
                "requested_output_path": str(output_path),
                "pipeline_config_path": str(config_path),
            },
        )

        pipeline.run(
            input_path=input_path,
            output_path=output_path,
            progress_callback=progress_callback,
            teardown_after_run=True,
        )

        actual_output_path = _resolve_existing_output_path(output_path)

        if not actual_output_path.exists():
            raise FileNotFoundError(
                f"Video pipeline completed, but output file was not found. "
                f"Expected: {output_path}"
            )

        update_job(
            job_id,
            metadata={
                "actual_output_path": str(actual_output_path),
            },
        )

        mark_job_completed(
            job_id,
            output_filename=actual_output_path.name,
            output_url=f"/api/v1/jobs/{job_id}/output",
            preview_url=f"/api/v1/jobs/{job_id}/output",
            message="Video processing completed.",
        )

    except Exception as e:
        logger.error(f"Video pipeline job {job_id} failed: {e}", exc_info=True)

        mark_job_failed(
            job_id,
            error_message=f"{str(e)}\n{traceback.format_exc()}",
        )

    finally:
        if pipeline is not None:
            try:
                pipeline.teardown(clear_pipeline=True)
            except Exception:
                logger.warning(
                    f"Failed to teardown video pipeline for job {job_id}.",
                    exc_info=True,
                )


async def process_video(request: VideoProcessRequest) -> JobResponse:
    """
    Start a video processing job.
    """
    input_path = Path(request.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    base_config_path = _resolve_pipeline_config_path(request)

    job_id = create_job(
        pipeline_id=request.pipeline_id,
        pipeline_name=base_config_path.stem,
        job_type=JobType.VIDEO,
        input_filename=input_path.name,
        metadata={
            "input_path": str(input_path),
            "pipeline_config_path": str(base_config_path),
            "overrides": request.overrides,
            "keep_original_audio": request.keep_original_audio,
            "output_format": request.output_format.value,
            "generate_preview": request.generate_preview,
        },
    )

    output_path = (
        Path(request.output_path)
        if request.output_path
        else _generate_video_output_path(
            input_path=input_path,
            job_id=job_id,
            output_format=request.output_format,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_config_path = _materialize_runtime_config(
        base_config_path=base_config_path,
        overrides=request.overrides,
        job_id=job_id,
    )

    update_job(
        job_id,
        output_filename=output_path.name,
        metadata={
            "requested_output_path": str(output_path),
            "runtime_config_path": str(runtime_config_path),
        },
    )

    asyncio.create_task(
        asyncio.to_thread(
            _execute_video_pipeline_sync,
            job_id,
            input_path,
            output_path,
            runtime_config_path,
        )
    )

    return JobResponse(
        job_id=job_id,
        status=_queued_status(),
        message="Video processing job queued.",
        pipeline_id=request.pipeline_id,
        job_type=JobType.VIDEO,
    )