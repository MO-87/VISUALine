import asyncio
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from visualine.api.schemas import (
    ImageBatchProcessRequest,
    ImageOutputFormat,
    ImageProcessRequest,
    JobResponse,
    JobStatus,
    JobType,
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


def _resolve_pipeline_config_path(request: ImageProcessRequest | ImageBatchProcessRequest) -> Path:
    """
    Resolve either:
    - pipeline_config_path: direct YAML path
    - pipeline_id: UI-friendly pipeline id/name
    """
    pipeline_ref = request.get_pipeline_reference()
    return resolve_pipeline_config_path(pipeline_ref)


def _generate_single_output_path(
    input_path: Path,
    job_id: str,
    output_format: ImageOutputFormat,
) -> Path:
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if output_format == ImageOutputFormat.SAME_AS_INPUT:
        suffix = input_path.suffix.lower()
    else:
        suffix = f".{output_format.value}"

    return job_dir / f"{input_path.stem}_processed{suffix}"


def _generate_batch_output_dir(job_id: str) -> Path:
    output_dir = OUTPUTS_DIR / job_id / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _materialize_runtime_config(
    base_config_path: Path,
    overrides: Dict[str, Any],
    job_id: str,
) -> Path:
    """
    Create a temporary YAML config with UI runtime overrides applied.
    """
    if not overrides:
        return base_config_path

    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid pipeline config: {base_config_path}")

    pipeline = config.get("pipeline", [])
    controls = config.get("ui", {}).get("controls", [])

    if not isinstance(pipeline, list):
        raise ValueError(f"Pipeline config has invalid 'pipeline' field: {base_config_path}")

    # 1. Map UI keys to node parameters using the 'controls' definition
    mapped_overrides = {}
    for key, value in overrides.items():
        # Check if this key is defined in UI controls
        control_def = next((c for c in controls if c.get("key") == key), None)
        
        if control_def:
            node_idx = control_def.get("node")
            param_name = control_def.get("param")
            
            if node_idx is not None and param_name:
                # Store as an explicit node override
                mapped_overrides[f"node.{node_idx}.{param_name}"] = value
                continue
        
        # If no control mapping found, keep as simple override
        mapped_overrides[key] = value

    # 2. Apply overrides to the pipeline
    for key, value in mapped_overrides.items():
        applied = False

        # Explicit syntax: nodes.0.prompt or node.0.prompt
        if key.startswith("nodes.") or key.startswith("node."):
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

        # Simple syntax: apply to every node that has this param
        for node_cfg in pipeline:
            params = node_cfg.setdefault("params", {})

            if key in params:
                params[key] = value
                applied = True

        if not applied:
            logger.warning(
                f"Override '{key}' did not match any mapping or node param in {base_config_path.name}."
            )

    temp_dir = Path(tempfile.gettempdir()) / "visualine_runtime_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = temp_dir / f"{job_id}_{base_config_path.name}"

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return runtime_config_path


def _execute_pipeline_sync(
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
        mark_job_processing(job_id, "Loading image pipeline...")

        config_loader = YamlConfigLoader()
        pipeline = PipelineManager(config_loader=config_loader)

        pipeline.load_pipeline(config_path)

        progress_callback = make_progress_callback(job_id)

        update_job(
            job_id,
            message="Processing image...",
            metadata={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "pipeline_config_path": str(config_path),
            },
        )

        pipeline.run(
            input_path=input_path,
            output_path=output_path,
            progress_callback=progress_callback,
            teardown_after_run=True,
        )

        mark_job_completed(
            job_id,
            output_filename=output_path.name,
            output_url=f"/api/v1/jobs/{job_id}/output",
            preview_url=f"/api/v1/jobs/{job_id}/output",
            message="Image processing completed.",
        )

    except Exception as e:
        logger.error(f"Image pipeline job {job_id} failed: {e}", exc_info=True)

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
                    f"Failed to teardown image pipeline for job {job_id}.",
                    exc_info=True,
                )


async def process_single_image(request: ImageProcessRequest) -> JobResponse:
    """
    Start a single-image processing job.
    """
    input_path = Path(request.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    base_config_path = _resolve_pipeline_config_path(request)

    job_id = create_job(
        pipeline_id=request.pipeline_id,
        pipeline_name=base_config_path.stem,
        job_type=JobType.IMAGE,
        input_filename=input_path.name,
        metadata={
            "input_path": str(input_path),
            "pipeline_config_path": str(base_config_path),
            "overrides": request.overrides,
        },
    )

    output_path = (
        Path(request.output_path)
        if request.output_path
        else _generate_single_output_path(
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
            "output_path": str(output_path),
            "runtime_config_path": str(runtime_config_path),
        },
    )

    asyncio.create_task(
        asyncio.to_thread(
            _execute_pipeline_sync,
            job_id,
            input_path,
            output_path,
            runtime_config_path,
        )
    )

    return JobResponse(
        job_id=job_id,
        status=_queued_status(),
        message="Image processing job queued.",
        pipeline_id=request.pipeline_id,
        job_type=JobType.IMAGE,
    )


async def process_image_batch(request: ImageBatchProcessRequest) -> JobResponse:
    """
    Start an image-directory batch processing job.
    """
    input_dir = Path(request.input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")

    base_config_path = _resolve_pipeline_config_path(request)

    job_id = create_job(
        pipeline_id=request.pipeline_id,
        pipeline_name=base_config_path.stem,
        job_type=JobType.BATCH,
        input_filename=input_dir.name,
        metadata={
            "input_dir": str(input_dir),
            "pipeline_config_path": str(base_config_path),
            "overrides": request.overrides,
        },
    )

    output_dir = Path(request.output_dir) if request.output_dir else _generate_batch_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = _materialize_runtime_config(
        base_config_path=base_config_path,
        overrides=request.overrides,
        job_id=job_id,
    )

    update_job(
        job_id,
        output_filename=output_dir.name,
        metadata={
            "output_dir": str(output_dir),
            "runtime_config_path": str(runtime_config_path),
        },
    )

    asyncio.create_task(
        asyncio.to_thread(
            _execute_pipeline_sync,
            job_id,
            input_dir,
            output_dir,
            runtime_config_path,
        )
    )

    return JobResponse(
        job_id=job_id,
        status=_queued_status(),
        message="Image batch processing job queued.",
        pipeline_id=request.pipeline_id,
        job_type=JobType.BATCH,
    )