import torch
import torch_tensorrt
import argparse
import logging
import os
import gc
from pathlib import Path

# Architecture imports
from visualine.models.archs.span_arch import SPAN
from visualine.models.loader import get_model_path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("UniversalCompiler")

def compile_model(model_type, tile_size, padding, batch_size):
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    block_size = tile_size + 2 * padding
    
    configs = {
        "span": {
            "file": "spanx4_ch48.pth",
            "out": "spanx4_ch48_trt.ts",
            "arch": lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=4)
        },
        "realesr-anime": {
            "file": "realesr-animevideov3.pth",
            "out": "realesr_anime_trt.ts",
            "arch": lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4)
        },
        "realesr-x4plus": {
            "file": "RealESRGAN_x4plus.pth",
            "out": "realesr_x4plus_trt.ts",
            "arch": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        }
    }

    cfg = configs.get(model_type)
    if not cfg:
        raise ValueError(f"Unknown model type: {model_type}")

    weights_path = get_model_path(cfg["file"])
    output_path = Path("weights") / cfg["out"]

    logger.info(f"--- Compiling {model_type.upper()} ---")
    logger.info(f"📏 Settings: Tile={tile_size}, Pad={padding}, Block={block_size}")
    logger.info(f"🚀 Batch Size: Optimal={batch_size}")
    
    # 1. Load Model
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    model = cfg["arch"]()
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Standard BasicSR weight extraction
    if 'params_ema' in state_dict:
        sd = state_dict['params_ema']
    elif 'params' in state_dict:
        sd = state_dict['params']
    else:
        sd = state_dict
        
    model.load_state_dict(sd, strict=True)
    model = model.cuda().half().eval()

    # 2. Trace
    logger.info(f"📸 Tracing with shape [1, 3, {block_size}, {block_size}]...")
    with torch.no_grad():
        example = torch.randn(1, 3, block_size, block_size).cuda().half()
        traced = torch.jit.trace(model, example)

    # 3. Torch-TRT Compile with Dynamic Batching
    logger.info("⚡ Running TensorRT Optimization (this may take 2-5 minutes)...")
    
    # We use a safety margin for max_shape
    max_batch = max(batch_size * 2, 48) 
    
    inputs = [torch_tensorrt.Input(
        min_shape=[1, 3, block_size, block_size],
        opt_shape=[batch_size, 3, block_size, block_size],
        max_shape=[max_batch, 3, block_size, block_size],
        dtype=torch.half
    )]
    
    try:
        trt_model = torch_tensorrt.compile(
            traced,
            ir="torchscript",
            inputs=inputs,
            enabled_precisions={torch.half},
            truncate_long_and_double=True,
            workspace_size=1 << 30
        )

        # 4. Save
        torch.jit.save(trt_model, str(output_path))
        logger.info(f"✅ SUCCESS! Saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ COMPILATION FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=64)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    compile_model(args.model, args.tile_size, args.padding, args.batch_size)
