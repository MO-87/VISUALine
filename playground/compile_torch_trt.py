import torch
import torch_tensorrt
import logging
from visualine.models.archs.span_arch import SPAN
from visualine.models.loader import get_model_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compile_span_trt(model_filename, output_ts_path, input_shape=(1, 3, 64, 64)):
    weights_path = get_model_path(model_filename)
    
    # 1. Load the standard PyTorch model
    logger.info(f"Loading weights from {weights_path}...")
    model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=4)
    state_dict = torch.load(weights_path, map_location='cpu')
    
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()

    # 2. Compile with Torch-TensorRT
    logger.info("Compiling model with Torch-TensorRT (Tracing)...")
    
    # Convert model to half for compilation
    model = model.half()

    # Trace the model first to bypass Python-heavy logic like update_params
    # We use a dummy input that matches our expected shape and format
    with torch.no_grad():
        example_input = torch.randn(input_shape).cuda().half().to(memory_format=torch.contiguous_format)
        traced_model = torch.jit.trace(model, example_input)

    # We use half precision (FP16) for massive speedup
    inputs = [
        torch_tensorrt.Input(
            shape=input_shape,
            dtype=torch.half,
            format=torch.contiguous_format
        )
    ]
    
    enabled_precisions = {torch.half}

    trt_gm = torch_tensorrt.compile(
        traced_model, 
        ir="torchscript",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        workspace_size=1 << 30, # 1GB
        truncate_long_and_double=True
    )

    # 3. Save the compiled TorchScript
    logger.info(f"Saving compiled model to {output_ts_path}...")
    torch.jit.save(trt_gm, output_ts_path)
    logger.info("Compilation successful!")

if __name__ == "__main__":
    model_name = "spanx4_ch48.pth"
    output_path = "weights/spanx4_ch48_trt.ts"
    
    # tile_size 64 + 2*padding 16 = 96
    compile_span_trt(model_name, output_path, input_shape=(1, 3, 96, 96))
