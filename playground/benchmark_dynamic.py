import os
import time
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

from visualine.nodes.upscale.span_node import SPANNode
from visualine.nodes.upscale.dynamic_upscale_node import DynamicUpscaleNode
from visualine.utils.file_io import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark(video_path: Path, node_class, config, output_dir: Path, tag: str, max_frames: int = None, skip_existing: bool = True):
    video_path = Path(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_{tag}.mp4"
    
    if skip_existing and output_path.exists():
        logger.info(f"Skipping {tag} for {video_path.name} (Output exists)")
        return None

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = node_class(config)
    node.setup(device)
    
    # Get output dimensions by running a dummy frame
    dummy_input = torch.zeros((1, 3, height, width), device=device)
    with torch.no_grad():
        dummy_output = node.process(dummy_input)
    out_h, out_w = dummy_output.shape[2], dummy_output.shape[3]
    
    vp = VideoProcessor()
    writer = vp.get_ffmpeg_writer(video_path, output_path, out_w, out_h, fps)
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc=f"Processing {video_path.name} ({tag})"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device).float()
            
            out_tensor = node.process(input_tensor)
            
            # RGB to BGR for writing
            out_frame = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            writer.stdin.write(out_frame.tobytes())
            
    end_time = time.time()
    duration = end_time - start_time
    avg_fps = total_frames / duration
    
    writer.stdin.close()
    writer.wait()
    cap.release()
    node.teardown()
    
    return {
        "video": video_path.name,
        "tag": tag,
        "duration": duration,
        "avg_fps": avg_fps,
        "total_frames": total_frames,
        "output_path": str(output_path)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to a specific video file or directory")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Force re-run all tests")
    args = parser.parse_args()

    output_dir = Path("data/output/benchmark/")
    
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            videos = [input_path]
        else:
            videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.mkv"))
    else:
        # Default fallback
        input_dir = Path("data/input/")
        videos = list(input_dir.glob("*.mp4"))
    
    if not videos:
        logger.error(f"No videos found to process at {args.input or 'data/input/'}")
        return

    # Load existing results if any
    results_file = output_dir / "benchmark_results.json"
    if results_file.exists() and not args.force:
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = []
    
    configs = [
        {
            "name": "Baseline_SPAN",
            "class": SPANNode,
            "params": {
                "model_filename": "spanx4_ch48.pth",
                "scale": 4,
                "feature_channels": 48,
                "fp16": True
            }
        },
        {
            "name": "Dynamic_T0.03_Optimized",
            "class": DynamicUpscaleNode,
            "params": {
                "model_filename": "spanx4_ch48.pth",
                "model_type": "span",
                "scale": 4,
                "threshold": 0.03,
                "tile_size": 64,
                "padding": 16,
                "refresh_interval": 60,
                "fp16": True,
                "edge_weight": 2.5,
                "dilation": 1,
                "persistence": 3,
                "global_threshold": 0.0001
            }
        },
        {
            "name": "Dynamic_T0.05_Aggressive",
            "class": DynamicUpscaleNode,
            "params": {
                "model_filename": "spanx4_ch48.pth",
                "model_type": "span",
                "scale": 4,
                "threshold": 0.05,
                "tile_size": 64,
                "padding": 16,
                "refresh_interval": 60,
                "fp16": True,
                "edge_weight": 3.0,
                "dilation": 1,
                "persistence": 2,
                "global_threshold": 0.0002
            }
        },
        {
            "name": "Dynamic_T0.03_TRT_Optimized",
            "class": DynamicUpscaleNode,
            "params": {
                "model_filename": "spanx4_ch48_trt.ts",
                "model_type": "span",
                "scale": 4,
                "threshold": 0.03,
                "tile_size": 64,
                "padding": 16,
                "refresh_interval": 60,
                "fp16": True,
                "batch_size": 1, # TRT model was exported with batch 1
                "edge_weight": 2.5,
                "dilation": 1,
                "persistence": 3,
                "global_threshold": 0.0001
            }
        }
    ]
    
    for video in videos:
        for cfg in configs:
            logger.info(f"Checking {cfg['name']} on {video.name}...")
            res = run_benchmark(video, cfg["class"], cfg["params"], output_dir, cfg["name"], 
                                max_frames=args.max_frames, skip_existing=not args.force)
            if res:
                # Replace or add new result
                results = [r for r in results if not (r["video"] == video.name and r["tag"] == cfg["name"])]
                results.append(res)
            
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("Benchmark update complete.")

if __name__ == "__main__":
    main()
