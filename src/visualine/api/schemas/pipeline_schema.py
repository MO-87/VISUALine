from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineInputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    GIF = "gif"
    BATCH = "batch"


class PipelineSpeed(str, Enum):
    FAST = "fast"
    MEDIUM = "medium"
    HEAVY = "heavy"


class PipelineControlType(str, Enum):
    TEXT = "text"
    NUMBER = "number"
    SLIDER = "slider"
    SELECT = "select"
    DROPDOWN = "dropdown"
    TOGGLE = "toggle"


class PipelineControl(BaseModel):
    key: str = Field(..., description="Override key sent back to the API")
    label: str = Field(..., description="Human-readable control label")
    type: PipelineControlType = Field(..., description="Frontend control type")

    default: Optional[Any] = None
    description: Optional[str] = None

    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    options: List[Any] = Field(default_factory=list)

    advanced: bool = Field(
        default=False,
        description="Whether this should be hidden in advanced settings",
    )

    node: Optional[int] = Field(
        None,
        description="Optional node index this control maps to",
    )

    param: Optional[str] = Field(
        None,
        description="Optional node param this control maps to",
    )


class PipelineNodeInfo(BaseModel):
    index: int
    class_path: str
    class_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelineSummary(BaseModel):
    id: str
    filename: str

    pipeline_name: str
    display_name: str

    category: str = "General"
    description: str = ""

    input_types: List[PipelineInputType] = Field(default_factory=list)
    speed: PipelineSpeed = PipelineSpeed.MEDIUM

    supports_prompt: bool = False
    is_hq: bool = False

    tags: List[str] = Field(default_factory=list)


class PipelineDetail(PipelineSummary):
    controls: List[PipelineControl] = Field(default_factory=list)
    nodes: List[PipelineNodeInfo] = Field(default_factory=list)
    raw_ui: Dict[str, Any] = Field(default_factory=dict)