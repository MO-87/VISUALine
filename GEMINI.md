# VISUALine: AI Visual Enhancement Suite

VISUALine is an optimized, node-based pipeline system for AI-powered visual enhancement, including upscaling, colorization, and frame interpolation. It is designed for high-performance inference using PyTorch, focusing on video processing through vectorized operations and intelligent resource management.

## Project Overview

- **Purpose**: Provide a modular and extensible suite for video/image enhancement.
- **Main Technologies**: 
    - **Language**: Python 3.10+
    - **Deep Learning**: PyTorch 2.5.0+, Torchvision 0.20.0+
    - **Inference Optimizations**: FP16 (Half Precision), `channels_last` memory format, CuDNN benchmarking.
    - **Architectures**: Real-ESRGAN, SPAN (Swift Parameter-free Attention Network), RIFE, DDColor.
    - **UI/API**: Streamlit for playgrounds/demos, FastAPI for web services.
- **Core Architecture**:
    - **Nodes (`src/visualine/nodes/`)**: Atomic processing units (e.g., `DynamicUpscaleNode`, `ColorizationNode`).
    - **Pipeline Manager (`src/visualine/core/pipeline_manager.py`)**: Orchestrates the flow of data through nodes.
    - **Resource Manager (`src/visualine/core/resource_manager.py`)**: Singleton handling LRU model caching and VRAM-aware eviction.
    - **Task Executer (`src/visualine/core/task_executer.py`)**: Compiles and executes the node graph.

## Building and Running

### Prerequisites
- **Python**: 3.10 to 3.12.
- **CUDA**: Recommended for GPU acceleration.
- **FFmpeg/FFprobe**: Required for video I/O.

### Installation
The project uses `uv` for dependency management.
```bash
# Sync dependencies
uv sync

# Install in editable mode
pip install -e .
```

### Running the Application
- **UI (Streamlit)**:
  ```bash
  visualine-ui
  # OR manually
  streamlit run src/visualine/app/main_app.py
  ```
- **API (FastAPI)**:
  ```bash
  visualine-api
  ```
- **Playground/Benchmarking**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  python playground/test_dynamic_upscale.py
  ```

## Development Conventions

- **Performance First**: Always prefer vectorized tensor operations over Python loops. Use FP16 (`.half()`) and `memory_format=torch.channels_last` where possible.
- **Node Interface**: New enhancement modules must inherit from `NodeBase` and implement `setup()`, `process()`, and `teardown()`.
- **Resource Safety**: Always use the `ResourceManager` to load models to prevent OOM errors and ensure proper cache management.
- **Video I/O**: Use `VideoProcessor` for FFmpeg-piped video writing to handle high-resolution output (e.g., 4K/5K) and ensure safe encoder fallbacks (NVENC -> libx264).
- **Code Style**: Adhere to PEP 8. `ruff` is used for linting and formatting (config in `pyproject.toml`).

## Directory Structure

- `src/visualine/core/`: Core engine and infrastructure.
- `src/visualine/nodes/`: Enhancement algorithms organized by type.
- `src/visualine/models/`: Architecture definitions and wrappers.
- `src/visualine/utils/`: Video, file, and metric utilities.
- `configs/pipeline_configs/`: YAML definitions for enhancement pipelines.
- `playground/`: Experimental scripts and benchmarking tools.
- `weights/`: Storage for AI model checkpoints (e.g., `.pth` files).
