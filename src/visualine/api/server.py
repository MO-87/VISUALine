import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from visualine.api.routes import image_router, system_router, video_router, jobs_router, pipelines_router
from visualine.core.logger import setup_logger
from visualine.core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = DATA_DIR / "outputs" / "api"
TEMP_DIR = DATA_DIR / "temp" / "api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifecycle.

    Startup:
    - Configure logging.
    - Create runtime directories.
    - Initialize ResourceManager.

    Shutdown:
    - Clear cached models/resources.
    """

    log_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        "filename": "visualine.log",
    }

    setup_logger(log_config, LOGS_DIR)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    resource_manager = ResourceManager(vram_limit_ratio=0.85)
    app.state.resource_manager = resource_manager

    logger.info("VISUALine API Server is starting up...")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"API outputs directory: {OUTPUTS_DIR}")
    logger.info(f"API temp directory: {TEMP_DIR}")

    try:
        yield

    finally:
        logger.info("VISUALine API Server is shutting down...")

        try:
            resource_manager.clear_cache()
        except Exception:
            logger.warning("Failed to clear ResourceManager cache cleanly.", exc_info=True)

        logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="VISUALine AI Suite API",
        description="Local backend for VISUALine Electron App",
        version="1.0.0",
        lifespan=lifespan,
    )

    # For local desktop/Electron development.
    # Avoid allow_origins=["*"] together with allow_credentials=True.
    allowed_origins: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8501",
        "file://",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"^(http://localhost:\d+|http://127\.0\.0\.1:\d+|file://.*)$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system_router)
    app.include_router(image_router)
    app.include_router(video_router)
    app.include_router(jobs_router)
    app.include_router(pipelines_router)

    # Mount static files to allow browser preview of local media
    # DANGEROUS: This gives the browser read access to the entire project root
    # but since it's a local-only backend on 127.0.0.1, it's acceptable for this application.
    app.mount("/media", StaticFiles(directory=str(PROJECT_ROOT)), name="media")

    @app.get("/")
    async def root():
        return {
            "message": "VISUALine API is running.",
            "docs": "/docs",
            "health": "/api/v1/system/health",
        }

    @app.get("/api/v1/status")
    async def status():
        return {
            "status": "online",
            "app": "VISUALine",
            "version": "1.0.0",
        }

    return app


app = create_app()


def start_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """
    Utility function to start the server programmatically.

    For Electron production/local app usage:
        reload=False

    For backend development:
        reload=True
    """
    uvicorn.run(
        "visualine.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    start_server(reload=False)
