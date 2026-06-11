import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from visualine.api.schemas.system_schema import OptimizeStatusResponse

logger = logging.getLogger(__name__)

# Find project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = PROJECT_ROOT / "weights"

is_compiling = False

async def get_optimized_status() -> OptimizeStatusResponse:
    """Check which models have optimized TRT engines available."""
    return OptimizeStatusResponse(
        span=(WEIGHTS_DIR / "spanx4_ch48_trt.ts").exists(),
        realesrgan=(WEIGHTS_DIR / "realesr_anime_trt.ts").exists(),
        rife=(WEIGHTS_DIR / "rife_trt.ts").exists(),
    )

async def run_optimization_generator(
    model_type: str,
    tile_size: int = 64,
    padding: int = 16,
    batch_size: int = 16
) -> AsyncGenerator[str, None]:
    """
    Execute the universal compiler script and yield output lines in real-time.
    """
    global is_compiling
    
    if is_compiling:
        yield "data: [ERROR] A compilation is already in progress.\n\n"
        return

    is_compiling = True
    script_path = PROJECT_ROOT / "playground" / "compile_universal.py"

    if not script_path.exists():
        is_compiling = False
        yield f"data: [ERROR] Optimization script not found: {script_path}\n\n"
        return

    logger.info(f"Starting optimization for {model_type} (Tile={tile_size}, Batch={batch_size})...")

    try:
        # Pass the new arguments
        args = [
            os.sys.executable, str(script_path),
            "--model", model_type,
            "--tile_size", str(tile_size),
            "--padding", str(padding),
            "--batch_size", str(batch_size)
        ]

        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:{PROJECT_ROOT}/src"}
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            clean_line = line.decode().strip()
            if clean_line:
                yield f"data: {clean_line}\n\n"

        await process.wait()
        
        if process.returncode == 0:
            yield f"data: [SUCCESS] {model_type.upper()} optimized successfully!\n\n"
            logger.info(f"Optimization for {model_type} succeeded.")
        else:
            yield f"data: [ERROR] Optimization process exited with code {process.returncode}.\n\n"
            logger.error(f"Optimization for {model_type} failed with code {process.returncode}.")

    except Exception as e:
        logger.exception(f"Unexpected error during optimization of {model_type}: {e}")
        yield f"data: [ERROR] System error: {str(e)}\n\n"
        
    finally:
        is_compiling = False
