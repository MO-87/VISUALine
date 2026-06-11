# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
# We use --system to install into the container's python environment
RUN uv pip install --system -e .
RUN uv pip install --system torch_tensorrt --extra-index-url https://download.pytorch.org/whl/cu124

# Create directory for weights and exports
RUN mkdir -p weights exports data/output/api

# Expose API port
EXPOSE 8000

# Entrypoint script to handle model compilation
RUN chmod +x scripts/entrypoint.sh
ENTRYPOINT ["scripts/entrypoint.sh"]

# Default command
CMD ["visualine-api"]
