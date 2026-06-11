# VISUALine: AI-Powered Visual Enhancement Suite

VISUALine is a modular, high-performance local AI video and image processing system designed for consumer GPUs. It transforms complex computer vision models into production-ready workflows with an intuitive, node-based architecture.

## 🚀 Key Features

*   **Dynamic AI Super-Resolution:** Motion-aware upscaling that selectively processes moving regions to maximize throughput. Supports **SPAN** and **Real-ESRGAN** (Anime & Standard).
*   **Prompt-Based Editing:** Control subject isolation using natural language via **GroundingDINO**.
*   **Cinematic Background Blur:** Professional depth-of-field effects using soft-masking or pixel-perfect **SAM2** tracking.
*   **Privacy Redaction:** Automated anonymization of faces and individuals in footage.
*   **Slow-Motion Interpolation:** Smooth frame generation powered by **RIFE v4.6**.
*   **Hardware Optimization:** Built-in **TensorRT Hub** for hardware-specific model compilation, significantly reducing inference latency.
*   **Dual-Mode Interface:** Runs as a native **Electron** desktop application or a robust **Web UI** with full local file system exploration and drag-and-drop support.

---

## 🛠️ Setup & Installation

### Prerequisites
*   **Python:** 3.10 or 3.11 (Recommended).
*   **Hardware:** NVIDIA GPU (RTX 30 series or higher recommended) with 4GB+ VRAM.
*   **OS:** Linux (Native or WSL2). Tested on Fedora, KDE Plasma, and Hyprland.
*   **Dependencies:** FFmpeg must be installed and available in your PATH.

### 1. Environment Setup
We recommend using `uv` for lightning-fast dependency management:

```bash
# Clone the repository
git clone https://github.com/your-repo/VISUALine.git
cd VISUALine

# Sync dependencies
uv sync

# Alternatively, using standard pip:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Hardware Acceleration (Optional)
To use the Optimization Hub (TensorRT), install the required libraries:

```bash
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3. Model Weights
Place your `.pth` or `.pt` model files in the `weights/` directory. The system will automatically detect them.

---

## 🖥️ Running the Application

### Browser Mode (Universal)
If you are on a system where Electron has compatibility issues, use the Browser Mode:

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
3.  Open `http://localhost:5173` in your browser.

### Native Desktop App
```bash
./run_app.sh
```

---

## 🏗️ Architecture

VISUALine is built with a focus on modularity and performance:

*   **Backend:** FastAPI (Python) serving a node-based execution engine.
*   **Frontend:** Vue 3 + Vite, styled for a modern dark-mode aesthetic.
*   **Core:**
    *   `PipelineManager`: Orchestrates the flow of data through nodes.
    *   `ResourceManager`: Handles LRU model caching and VRAM protection.
    *   `VideoProcessor`: High-speed FFmpeg-piped I/O with audio preservation.

---

## 📄 Documentation
For detailed information on recent updates, normalization fixes, and technical architecture, see [docs/ENHANCEMENTS_JUNE_2026.md](docs/ENHANCEMENTS_JUNE_2026.md).

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
