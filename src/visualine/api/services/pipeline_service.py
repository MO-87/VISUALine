import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from visualine.api.schemas import (
    PipelineControl,
    PipelineDetail,
    PipelineInputType,
    PipelineNodeInfo,
    PipelineSpeed,
    PipelineSummary,
)

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """
    Robustly find project root from this file location.

    Expected root contains:
    - pyproject.toml
    - configs/
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent

    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs" / "pipeline_configs"


def list_pipelines() -> List[PipelineSummary]:
    """
    List available user-facing pipeline YAML configs as UI-friendly summaries.

    Only configs with a top-level `ui:` block are shown in the frontend.
    Internal/test configs stay hidden.
    """
    if not CONFIGS_DIR.exists():
        logger.warning(f"Pipeline configs directory does not exist: {CONFIGS_DIR}")
        return []

    summaries: List[PipelineSummary] = []

    for config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            ui = config.get("ui")

            if not isinstance(ui, dict):
                continue

            if ui.get("show_in_ui") is False:
                continue

            detail = get_pipeline_detail(config_path.stem)
            summaries.append(PipelineSummary(**detail.model_dump()))

        except Exception as e:
            logger.warning(
                f"Failed to load pipeline config {config_path}: {e}",
                exc_info=True,
            )

    return summaries


def get_pipeline_detail(pipeline_id: str) -> PipelineDetail:
    """
    Load one pipeline config and return full UI metadata.
    """
    config_path = resolve_pipeline_config_path(pipeline_id)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")

    pipeline_name = str(config.get("pipeline_name", config_path.stem))
    ui = config.get("ui", {}) or {}

    if not isinstance(ui, dict):
        ui = {}

    display_name = str(
        ui.get("display_name")
        or _prettify_name(pipeline_name)
        or _prettify_name(config_path.stem)
    )

    category = str(ui.get("category", _infer_category(config_path.stem)))
    description = str(ui.get("description", _infer_description(config_path.stem)))

    input_types = _parse_input_types(ui.get("input_types", _infer_input_types(config_path.stem)))
    speed = _parse_speed(ui.get("speed", _infer_speed(config_path.stem)))

    controls = _parse_controls(ui.get("controls", []))
    nodes = _parse_nodes(config.get("pipeline", []))

    supports_prompt = bool(
        ui.get("supports_prompt", _infer_supports_prompt(nodes, controls))
    )

    is_hq = bool(ui.get("is_hq", "sam2" in config_path.stem.lower()))

    tags = ui.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    return PipelineDetail(
        id=config_path.stem,
        filename=config_path.name,
        pipeline_name=pipeline_name,
        display_name=display_name,
        category=category,
        description=description,
        input_types=input_types,
        speed=speed,
        supports_prompt=supports_prompt,
        is_hq=is_hq,
        tags=[str(tag) for tag in tags],
        controls=controls,
        nodes=nodes,
        raw_ui=ui,
    )


def resolve_pipeline_config_path(pipeline_id: str) -> Path:
    """
    Resolve:
    - 01_cinematic_prompt_blur
    - 01_cinematic_prompt_blur.yaml
    - absolute/path/to/config.yaml
    """
    candidate = Path(pipeline_id)

    if candidate.exists():
        return candidate

    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")

    resolved = CONFIGS_DIR / candidate.name

    if not resolved.exists():
        raise FileNotFoundError(
            f"Pipeline config not found for '{pipeline_id}'. Tried: {resolved}"
        )

    return resolved


def _parse_controls(raw_controls: Any) -> List[PipelineControl]:
    if not isinstance(raw_controls, list):
        return []

    controls: List[PipelineControl] = []

    for item in raw_controls:
        if not isinstance(item, dict):
            continue

        try:
            controls.append(PipelineControl(**item))
        except Exception as e:
            logger.warning(f"Invalid pipeline UI control skipped: {item} | {e}")

    return controls


def _parse_nodes(raw_pipeline: Any) -> List[PipelineNodeInfo]:
    if not isinstance(raw_pipeline, list):
        return []

    nodes: List[PipelineNodeInfo] = []

    for index, node_cfg in enumerate(raw_pipeline):
        if not isinstance(node_cfg, dict):
            continue

        class_path = str(node_cfg.get("class", ""))
        class_name = class_path.split(".")[-1] if class_path else ""

        params = node_cfg.get("params", {})
        if not isinstance(params, dict):
            params = {}

        nodes.append(
            PipelineNodeInfo(
                index=index,
                class_path=class_path,
                class_name=class_name,
                params=params,
            )
        )

    return nodes


def _parse_input_types(value: Any) -> List[PipelineInputType]:
    if not isinstance(value, list):
        return []

    output: List[PipelineInputType] = []

    for item in value:
        try:
            output.append(PipelineInputType(str(item)))
        except Exception:
            logger.warning(f"Unknown pipeline input type ignored: {item}")

    return output


def _parse_speed(value: Any) -> PipelineSpeed:
    try:
        return PipelineSpeed(str(value))
    except Exception:
        return PipelineSpeed.MEDIUM


def _infer_supports_prompt(
    nodes: List[PipelineNodeInfo],
    controls: List[PipelineControl],
) -> bool:
    for control in controls:
        if control.key == "prompt" or control.param == "prompt":
            return True

    for node in nodes:
        if "prompt" in node.params:
            return True

    return False


def _infer_input_types(pipeline_id: str) -> List[str]:
    lower = pipeline_id.lower()

    if "image" in lower or "upscale" in lower:
        return ["image", "video", "gif"]

    return ["video", "gif"]


def _infer_speed(pipeline_id: str) -> str:
    lower = pipeline_id.lower()

    if "sam2" in lower or "span" in lower or "upscale" in lower:
        return "heavy"

    if "rife" in lower or "privacy" in lower or "cinematic" in lower:
        return "medium"

    return "fast"


def _infer_category(pipeline_id: str) -> str:
    lower = pipeline_id.lower()

    if "privacy" in lower or "redaction" in lower:
        return "Privacy"

    if "enhancement" in lower or "upscale" in lower:
        return "Restoration"

    if "reframe" in lower:
        return "Social Media"

    if "slow" in lower or "rife" in lower:
        return "Motion"

    if "cinematic" in lower or "blur" in lower:
        return "AI Editing"

    return "General"


def _infer_description(pipeline_id: str) -> str:
    lower = pipeline_id.lower()

    if "cinematic" in lower:
        return "Keep a prompted subject sharp while blurring the background."

    if "privacy" in lower:
        return "Automatically redact faces, people, or sensitive regions."

    if "enhancement" in lower:
        return "Improve old or low-quality footage with denoising, contrast, and sharpening."

    if "reframe" in lower:
        return "Convert videos into vertical social-media format while following the subject."

    if "rife" in lower or "slow" in lower:
        return "Create smoother slow-motion footage using AI frame interpolation."

    return ""


def _prettify_name(name: str) -> str:
    name = name.replace(".yaml", "").replace(".yml", "")
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(part for part in name.split() if not part.isdigit())

    return name.title()