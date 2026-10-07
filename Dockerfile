# Dockerfile for Runpod Serverless Unified Worker (CUDA 12.1 + PyTorch + Ollama LLaVA + Hunyuan3D)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA="1" \
    HF_HUB_DISABLE_XET="1" \
    HF_HUB_ENABLE_HF_TRANSFER="0" \
    PYTHONPATH="/app:/app/3dgen/Hunyuan3D-2-main"

# Install Python 3.10 and necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglib2.0-0 \
    libopengl0 \
    libegl1 \
    build-essential \
    ninja-build \
    procps \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/cuda/targets/x86_64-linux/lib/*_static.a || true

# Set python3 alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ollama CLI and daemon
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pre-download LLaVA 7B model into Docker image layer
RUN ollama serve & \
    PID=$! && \
    for i in $(seq 1 30); do curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && break || sleep 1; done && \
    ollama pull llava:7b && \
    kill $PID && \
    wait $PID || true

WORKDIR /app

# Copy dependency definitions and install Python packages
COPY requirements-runpod.txt /app/requirements-runpod.txt
RUN pip3 install --no-cache-dir -r /app/requirements-runpod.txt

# Copy Hunyuan3D-2 codebase and compile C++/CUDA native rasterizers
COPY 3dgen/Hunyuan3D-2-main /app/3dgen/Hunyuan3D-2-main

RUN pip3 install --no-cache-dir --no-deps -e /app/3dgen/Hunyuan3D-2-main && \
    cd /app/3dgen/Hunyuan3D-2-main/hy3dgen/texgen/custom_rasterizer && \
    python3 setup.py install && \
    cd /app/3dgen/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer && \
    python3 setup.py install

# Copy application files
COPY runpod_handler.py /app/runpod_handler.py
COPY test_handler.py /app/test_handler.py
COPY preload_models.py /app/preload_models.py
COPY start.sh /app/start.sh

# Pre-download rembg and Hunyuan3D shape model into Docker image layer
RUN python3 /app/preload_models.py

RUN chmod +x /app/start.sh

# Run entrypoint script
CMD ["/app/start.sh"]
