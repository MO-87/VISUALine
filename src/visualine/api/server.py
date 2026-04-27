import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from visualine.core.pipeline_manager import PipelineManager
from visualine.core.config_loader import YamlConfigLoader

app = FastAPI(title="VISUALine API", description="AI Visual Enhancement Suite")

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "pipeline_configs"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "api"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Core
config_loader = YamlConfigLoader()
pipeline_manager = PipelineManager(config_loader=config_loader)

class PipelineRunRequest(BaseModel):
    pipeline_config: str = "test_dynamic_upscale.yaml"

@app.get("/")
def read_root():
    return {"status": "online", "message": "VISUALine API is active"}

@app.get("/pipelines")
def list_pipelines():
    configs = [p.name for p in CONFIGS_DIR.glob("*.yaml")]
    return {"available_pipelines": configs}

@app.post("/upscale/video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pipeline_config: str = "test_dynamic_upscale.yaml"
):
    config_path = CONFIGS_DIR / pipeline_config
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline config not found")

    # 1. Save uploaded file to temp
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = Path(tmp_in.name)

    # 2. Prepare output path
    output_filename = f"processed_{file.filename}"
    output_path = DATA_OUTPUT_DIR / output_filename

    # 3. Run Pipeline
    try:
        pipeline_manager.load_pipeline(config_path)
        pipeline_manager.run(input_path, output_path)
    except Exception as e:
        if input_path.exists(): os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    # 4. Cleanup input
    if input_path.exists():
        os.remove(input_path)

    return {
        "message": "Processing complete",
        "output_file": output_filename,
        "download_url": f"/download/{output_filename}"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = DATA_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
