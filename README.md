# VISUALine: A Modular System for Local AI Video Processing on Consumer GPUs

VISUALine is a modular, high-performance local AI video and image processing system designed for consumer GPUs. It transforms state-of-the-art computer vision models into production-ready workflows with an intuitive, node-based architecture.

## Key Features

*   **Dynamic AI Super-Resolution (VSR):** A motion-aware upscaling engine that selectively processes regions with movement, maximizing throughput and reducing power consumption. Supports **SPAN** and **Real-ESRGAN** variants.
*   **Prompt-Based Subject Isolation:** Leverage **GroundingDINO** to identify and mask subjects using natural language prompts (e.g., "the person on the left", "red car").
*   **Cinematic Background Blur:** Professional depth-of-field effects using optimized soft-masking or pixel-perfect **SAM2** (Segment Anything 2) video tracking.
*   **Privacy Redaction Suite:** Automated anonymization of sensitive visual data, including faces and specific individuals.
*   **Slow-Motion Interpolation:** Ultra-smooth frame generation powered by a customized **RIFE v4.6** implementation.
*   **Hardware-Aware Optimization:** Built-in **TensorRT Hub** for compiling AI architectures into hardware-specific engines, significantly reducing inference latency on NVIDIA hardware.
*   **Hybrid Interface Architecture:** Native **Electron** desktop application paired with a robust **Universal Web UI** featuring local file system exploration and advanced drag-and-drop support.

---

## Setup & Installation

### Prerequisites
*   **Python:** 3.10 or 3.11.
*   **Hardware:** NVIDIA GPU (RTX 30 series or higher recommended) with 4GB+ VRAM.
*   **OS:** Linux (Native or WSL2). Optimized for Wayland (Hyprland/KDE Plasma).
*   **System Tools:** FFmpeg must be installed in your system PATH.

### 1. Environment Setup
We recommend using [uv](https://github.com/astral/uv) for dependency management:

```bash
# Clone the repository
git clone git@github.com:MO-87/VISUALine.git
cd VISUALine

# Sync dependencies
uv sync

# Alternatively, using standard pip:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Hardware Acceleration (Optional)
To use the TensorRT Optimization features:

```bash
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3. Model Weights
Place your AI model files (`.pth`, `.pt`, `.ts`) in the `weights/` directory. The system automatically detects and loads the appropriate wrappers.

---

## Running VISUALine

### Universal Web UI (Recommended for Linux/WSL)
Best for users on Hyprland, KDE, or WSL2 where Electron display protocols can vary:

1.  **Start the Backend:**
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python src/visualine/api/server.py
    ```
2.  **Start the Frontend:**
    ```bash
    cd src/visualine/app
    npm install
    npm run dev
    ```
3.  **Access:** Open `http://localhost:5173` in your browser.

### Native Desktop Application
```bash
cd src/visualine/app
npm run dev
```

---

## Architecture

VISUALine is built on a clean, scalable stack:

*   **Execution Engine:** Python-based node graph that manages model lifecycles and vectorized tensor operations.
*   **Resource Management:** Singleton `ResourceManager` for LRU model caching and VRAM-aware eviction.
*   **Video I/O:** High-performance FFmpeg pipes with support for rational framerates and audio stream preservation.
*   **Frontend:** Vue 3 + Vite, utilizing a modern dark-mode studio aesthetic.

---

## Documentation
For technical implementation details, normalization logic, and the June 2026 performance benchmarks, refer to:
*   [docs/Bachelor_Thesis.pdf](docs/Mohammed_Marzouk_Bachelor_Thesis.pdf)

## License
This project is licensed under the MIT License.
