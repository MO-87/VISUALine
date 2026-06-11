#!/bin/bash

echo "--- VISUALine Launcher ---"

# Check for NVIDIA Container Toolkit
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo "❌ ERROR: NVIDIA Container Toolkit not found in Docker."
    echo "Please install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
fi

echo "📦 Building and starting containers..."
docker-compose up --build -d

echo "✅ VISUALine is now running!"
echo "📍 API: http://localhost:8000"
echo "🖥️  Test UI: Open 'playground/dummy_ui.html' in your browser."
echo ""
echo "To see live logs, run: docker logs -f visualine-api"
