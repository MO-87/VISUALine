import os
import time
import json
import torch
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Optional

from visualine.core.config_loader import YamlConfigLoader
from visualine.core.pipeline_manager import PipelineManager
from visualine.core.resource_manager import ResourceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = Path("data/output/benchmark_auto")
RESULTS_CSV = OUTPUT_DIR / "benchmark_results.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "benchmark_checkpoint.json"
INPUT_DIR = Path("data/inputs/video")

TEST_MATRIX = [
    {
        "config": "configs/pipeline_configs/03_old_video_enhancement_span_x4.yaml",
        "video": "caltech_ml_lec.mp4",
        "params": {"model_filename": "spanx4_ch48.pth"},
        "id": "enhance_span"
    },
    {
        "config": "configs/pipeline_configs/05_slow_motion_rife_x2.yaml",
        "video": "high_motion_nature.mp4",
        "params": {},
        "id": "rife_x2"
    }
]

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"completed": []}
    return {"completed": []}

def save_checkpoint(completed_ids):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"completed": completed_ids}, f)

def run_test(test_item: Dict[str, Any], max_frames: Optional[int] = None):
    config_path = Path(test_item["config"])
    video_path = INPUT_DIR / test_item["video"]
    output_path = OUTPUT_DIR / f"{test_item['id']}_{test_item['video']}"
    
    logger.info(f"Running test {test_item['id']} on {test_item['video']}...")
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.perf_counter()
    
    try:
        loader = YamlConfigLoader()
        pm = PipelineManager(config_loader=loader)
        
        # Load config and override params if needed
        config = loader.load(config_path)
        for node_cfg in config.get("pipeline", []):
            if "params" not in node_cfg:
                node_cfg["params"] = {}
            node_cfg["params"]["batch_size"] = 1  # Force batch_size=1 to prevent OOM
            
            if test_item.get("params"):
                for k, v in test_item["params"].items():
                    if k in node_cfg["params"]:
                        node_cfg["params"][k] = v
                        logger.info(f"Overriding param '{k}' with '{v}' in node '{node_cfg.get('class')}'")
        
        pm.load_pipeline_from_dict(config)
        pm.run(video_path, output_path, max_frames=max_frames)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_vram = 0
            
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        
        effective_fps = frame_count / total_time
        
        # Verify output exists
        out_size = 0
        if output_path.exists():
            out_size = output_path.stat().st_size / (1024**2)
        else:
            # Maybe it added a suffix like .mp4 if it wasn't there?
            # VideoProcessor might change extension.
            possible_outputs = list(OUTPUT_DIR.glob(f"{test_item['id']}_{test_item['video']}*"))
            if possible_outputs:
                output_path = possible_outputs[0]
                out_size = output_path.stat().st_size / (1024**2)

        result = {
            "run_id": test_item["id"],
            "workflow": config.get("pipeline_name", test_item["id"]),
            "input": test_item["video"],
            "resolution": f"{int(width)}x{int(height)}",
            "duration": f"{frame_count/fps:.2f}s" if fps > 0 else "0s",
            "fps": f"{fps:.2f}",
            "total_time": f"{total_time:.2f}",
            "effective_fps": f"{effective_fps:.2f}",
            "peak_vram": f"{peak_vram:.2f}",
            "output_size": f"{out_size:.2f}",
            "prompt": str(test_item["params"].get("prompt", "")),
            "status": "success",
            "notes": ""
        }
        
        pm.teardown(clear_pipeline=True)
        ResourceManager().clear_cache()
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

    except Exception as e:
        logger.error(f"Test {test_item['id']} failed: {e}", exc_info=True)
        try:
            pm.teardown(clear_pipeline=True)
        except:
            pass
        ResourceManager().clear_cache()
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return {
            "run_id": test_item["id"],
            "workflow": test_item["id"],
            "input": test_item["video"],
            "status": "failed",
            "notes": str(e)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Only process a few frames for testing")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    
    file_exists = RESULTS_CSV.exists()
    
    fieldnames = [
        "run_id", "workflow", "input", "resolution", "duration", "fps", 
        "total_time", "effective_fps", "peak_vram", "output_size", 
        "prompt", "status", "notes"
    ]

    # Use 'a' to append results
    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        for test_item in TEST_MATRIX:
            if test_item["id"] in checkpoint["completed"]:
                logger.info(f"Skipping {test_item['id']} as it is already completed.")
                continue
            
            max_frames = 10 if args.quick else None
            result = run_test(test_item, max_frames=max_frames)
            if result:
                # Ensure all fieldnames are present in result
                row = {k: result.get(k, "") for k in fieldnames}
                writer.writerow(row)
                csvfile.flush() # Ensure it's written in case of crash later
                
                if result.get("status") == "success":
                    checkpoint["completed"].append(test_item["id"])
                    save_checkpoint(checkpoint["completed"])

    logger.info("All tests completed.")

if __name__ == "__main__":
    main()
