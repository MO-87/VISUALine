import os
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
from basicsr.archs.srvgg_arch import SRVGGNetCompact

# Settings
MODEL_FILENAME = "realesr-animevideov3.pth"
WEIGHTS_DIR = Path("weights")
OUTPUT_ONNX = WEIGHTS_DIR / "realesr_anime.onnx"
OUTPUT_TRT = WEIGHTS_DIR / "realesr_anime.engine"
TILE_SIZE = 64 # Optimized for TRT
BLOCK_SIZE = TILE_SIZE + 16*2 # 64 + 32 = 96
BATCH_SIZE = 32

def export_onnx():
    print("🚀 Loading PyTorch model...")
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    
    weight_path = WEIGHTS_DIR / MODEL_FILENAME
    loadnet = torch.load(str(weight_path), map_location='cpu')
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval().cuda()

    print(f"📦 Exporting to ONNX: {OUTPUT_ONNX}")
    dummy_input = torch.randn(BATCH_SIZE, 3, BLOCK_SIZE, BLOCK_SIZE).cuda()
    
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_ONNX),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print("✅ ONNX Export Complete.")

def build_trt():
    print("🏗️ Building TensorRT Engine (this takes a few minutes)...")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    config = builder.create_builder_config()
    # 3050 Laptop has ~4GB VRAM. Let's use 1GB for workspace.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16) # Enable FP16

    with open(str(OUTPUT_ONNX), 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Optimization Profile for dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, BLOCK_SIZE, BLOCK_SIZE), (BATCH_SIZE, 3, BLOCK_SIZE, BLOCK_SIZE), (BATCH_SIZE, 3, BLOCK_SIZE, BLOCK_SIZE))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    with open(str(OUTPUT_TRT), "wb") as f:
        f.write(engine_bytes)
    print(f"✅ TensorRT Engine saved to: {OUTPUT_TRT}")

if __name__ == "__main__":
    if not OUTPUT_ONNX.exists():
        export_onnx()
    build_trt()
