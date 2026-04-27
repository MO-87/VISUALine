#!/bin/bash
set -e

echo "--- VISUALine Boot Sequence ---"

# 1. Check for TRT Model
TRT_MODEL="weights/spanx4_ch48_trt.ts"

if [ ! -f "$TRT_MODEL" ]; then
    echo "Optimized TRT model not found. Starting one-time compilation..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python3.11 playground/compile_torch_trt.py
else
    echo "Optimized TRT model found. Skipping compilation."
fi

echo "--- Starting Application ---"
# Execute the main command (e.g., visualine-api)
exec "$@"
