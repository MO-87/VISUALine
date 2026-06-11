# VISUALine: Recent System Enhancements & Fixes (June 2026)

This document summarizes the major architectural improvements and feature integrations implemented during the June 2026 development phase. These updates focus on hardware optimization, browser-mode usability, and robustness of the AI pipeline.

## 1. Dynamic AI Super-Resolution Integration
The **Dynamic Upscaling** workflow is now fully integrated into the core pipeline and UI. This system utilizes temporal consistency and motion-masking to skip processing on static frame regions, significantly increasing performance on consumer GPUs.

*   **Vectorized Tiling:** Replaced Python-based loops with optimized PyTorch `unfold` and `permute` operations for near-zero overhead tiling and stitching.
*   **Color-Space Correction:** Resolved "green pixelation" issues by ensuring the entire pipeline operates in RGB space and correctly restoring the **RGB Mean** subtracted by models like SPAN.
*   **User Controls:** Added real-time sliders for **Motion Sensitivity** and dropdowns for **Model Selection** (SPAN, Real-ESRGAN Anime, Real-ESRGAN 4x Plus).

## 2. Hardware Optimization Hub (TensorRT)
A new dedicated module for hardware-specific model compilation has been added, allowing users to generate high-performance TensorRT engines directly from the UI.

*   **Universal Compiler:** Refactored `playground/compile_universal.py` to support customizable **Tile Size**, **Padding**, and **Dynamic Batching**.
*   **Live Log Streaming:** Implemented a Server-Sent Events (SSE) backend to stream compilation logs to a professional-grade console in the frontend.
*   **Auto-Detection:** The `DynamicUpscaleNode` now automatically detects and switches to optimized `.ts` files when they are available in the `weights/` directory.

## 3. Full Browser Mode Support
To ensure usability on Linux environments (Hyprland/KDE) where Electron may face launch issues, the web-based version has been upgraded to a first-class citizen.

*   **Local File Browser:** Built a custom `BrowserFileExplorer.vue` that interacts with a new `/api/v1/system/explore` endpoint.
*   **Path Resolution Shortcuts:** Added one-click shortcuts for **Home** (`~`) and **Project Videos** to simplify navigation in browser-only mode.
*   **Recursive Drag-and-Drop:** Implemented a backend resolver that uses recursive file searching to find the absolute paths of media files dropped into the browser window.
*   **Static Media Serving:** Mounted the project root to a secure `/media` static route, enabling full-resolution video previews in any standard web browser.

## 4. Stability & Robustness Fixes
*   **GroundingDINO Patch:** Applied a "Pure PyTorch" fallback to the GroundingDINO detector, allowing it to run on systems missing the CUDA Toolkit compiler (`nvcc`).
*   **VRAM Management:** Enforced aggressive memory clearing (`gc.collect` + `torch.cuda.empty_cache`) and `batch_size=1` limits for GPUs with 4GB of VRAM (like the RTX 3050 Laptop).
*   **RIFE Weights:** Sourced and integrated the correct 15-channel `flownet.pkl` required by the customized `IFNet_HDv3` architecture.
*   **Frame Count Fix:** Updated the FFmpeg pipeline to use exact fraction strings for framerates, resolving the "1 dropped frame" bug during video muxing.

---
*Documentation generated for VISUALine AI Suite.*
