import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from visualine.core.logger import setup_logger
from visualine.core.resource_manager import ResourceManager
from visualine.api.routes import system_router, image_router, video_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ## --- STARTUP ---
    log_dir = Path("logs")
    log_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        "filename": "visualine.log"
    }
    setup_logger(log_config, log_dir)
    
    rm = ResourceManager(vram_limit_ratio=0.85)
    logger.info("VISUALine API Server is starting up...")
    
    yield
    
    ## --- SHUTDOWN ---
    logger.info("VISUALine API Server is shutting down...")
    rm.clear_cache()
    logger.info("Shutdown complete.")

app = FastAPI(
    title="VISUALine AI Suite API",
    description="Local backend for VISUALine Electron App",
    version="1.0.0",
    lifespan=lifespan
)

## CORS Configuration (Crucial for Electron)
## Electron's frontend will likely run on localhost:3000 or from a file:// protocol
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  ## allow all for local desktop app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(image_router)
app.include_router(video_router)

## Health Check Route
@app.get("/")
async def root():
    return {"message": "VISUALine API is running."}

def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Utility function to start the server programmatically."""
    uvicorn.run("visualine.api.server:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    start_server(reload=True)