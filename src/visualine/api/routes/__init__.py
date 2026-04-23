from .system import router as system_router
from .image import router as image_router
from .video import router as video_router

__all__ = [
    "system_router",
    "image_router",
    "video_router",
]