# Dockerfile for Runpod Serverless Unified Worker (CUDA 12.1 + PyTorch + Ollama LLaVA + Hunyuan3D)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA="1" \
    PYTHONPATH="/app:/app/3dgen/Hunyuan3D-2-main"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI and daemon
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pre-download LLaVA 7B model into Docker image layer
RUN ollama serve & \
    PID=$! && \
    sleep 5 && \
    ollama pull llava:7b && \
    kill $PID && \
    wait $PID || true

WORKDIR /app

# Copy dependency definitions and install Python packages
COPY requirements-runpod.txt /app/requirements-runpod.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements-runpod.txt

# Copy Hunyuan3D-2 codebase and compile C++/CUDA native rasterizers
COPY 3dgen/Hunyuan3D-2-main /app/3dgen/Hunyuan3D-2-main

RUN pip install --no-cache-dir -e /app/3dgen/Hunyuan3D-2-main && \
    cd /app/3dgen/Hunyuan3D-2-main/hy3dgen/texgen/custom_rasterizer && \
    python setup.py install && \
    cd /app/3dgen/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer && \
    python setup.py install

# Copy application files
COPY runpod_handler.py /app/runpod_handler.py
COPY test_handler.py /app/test_handler.py
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

# Run entrypoint script
CMD ["/app/start.sh"]
