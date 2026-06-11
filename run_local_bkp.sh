#!/bin/bash
echo "--- VISUALine Backend Launcher ---"

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
VENV_PYTHON="/home/theodoros/graduation/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ ERROR: Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

echo "🚀 Starting FastAPI server on port 8000..."
# Killing any process on 8000 just in case
fuser -k 8000/tcp 2>/dev/null

$VENV_PYTHON src/visualine/api/server.py
