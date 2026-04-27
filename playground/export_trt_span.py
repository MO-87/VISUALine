import tensorrt as trt
import torch
import os
from pathlib import Path
from visualine.models.archs.span_arch import SPAN
from visualine.models.loader import get_model_path

def export_onnx(model, onnx_path, input_shape=(1, 3, 64, 64)):
    dummy_input = torch.randn(input_shape).cuda()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Fixed shape for max performance on TRT
    )
    print(f"ONNX model exported to {onnx_path}")

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Increase workspace memory for better optimization
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("Building TensorRT engine (this may take a few minutes)...")
    plan = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(plan)
    print(f"TensorRT engine saved to {engine_path}")

def main():
    model_filename = "spanx4_ch48.pth"
    weights_path = get_model_path(model_filename)
    onnx_path = "weights/spanx4_ch48.onnx"
    engine_path = "weights/spanx4_ch48.engine"

    # Load model
    model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=4)
    state_dict = torch.load(weights_path, map_location='cpu')
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()

    # Export
    export_onnx(model, onnx_path)
    build_engine(onnx_path, engine_path)

if __name__ == "__main__":
    main()
