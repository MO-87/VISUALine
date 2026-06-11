import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from visualine.api.schemas import PipelineDetail, PipelineSummary
from visualine.api.services.pipeline_service import (
    get_pipeline_detail,
    list_pipelines,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/pipelines",
    tags=["Pipelines"],
)


@router.get(
    "",
    response_model=List[PipelineSummary],
)
async def list_pipelines_endpoint() -> List[PipelineSummary]:
    """
    List available VISUALine pipeline workflows.

    Used by the frontend to render the workflow gallery.
    """
    return list_pipelines()


@router.get(
    "/{pipeline_id}",
    response_model=PipelineDetail,
)
async def get_pipeline_detail_endpoint(pipeline_id: str) -> PipelineDetail:
    """
    Return full metadata for a single pipeline.

    Used by the frontend to render workflow controls dynamically.
    """
    try:
        return get_pipeline_detail(pipeline_id)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.error(
            f"Failed to load pipeline detail for {pipeline_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load pipeline metadata.",
        ) from e