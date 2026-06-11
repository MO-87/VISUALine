# VISUALine Deployment Guide (Linux + NVIDIA)

This folder contains everything needed to run VISUALine with high-performance hardware acceleration on your NVIDIA GPU.

## 🚀 Prerequisites
1.  **NVIDIA GPU** (RTX 20-series or newer recommended).
2.  **Docker** and **Docker Compose** installed.
3.  **NVIDIA Container Toolkit** installed.
    *   [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 🛠️ One-Click Setup
Simply run:
```bash
./run_app.sh
```

## 🎮 How to use the Test UI
1.  Once the app is running, the API is available at `http://localhost:8000`.
2.  Open the file `playground/dummy_ui.html` in your browser.
3.  Click **[ Optimize SPAN for My GPU ]**.
    *   This will take about 1-2 minutes. It is tailoring the AI model specifically for your exact graphics card.
4.  Once the status changes to **READY (TRT)**, the app is running at maximum speed.

## 📁 Project Structure
*   `src/`: Backend logic.
*   `weights/`: AI Model storage (optimized models will be saved here).
*   `configs/`: Pipeline settings.
*   `data/`: Put your input videos here.
