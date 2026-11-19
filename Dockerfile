# syntax=docker/dockerfile:1.4
# Dockerfile for Runpod Serverless Hunyuan3D Worker (CUDA 12.1 + PyTorch)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA="1" \
    HF_HUB_DISABLE_XET="1" \
    HF_HUB_ENABLE_HF_TRANSFER="0" \
    PYTHONPATH="/app:/app/3d_generative/Hunyuan3D-2-main" \
    RUNPOD_VOLUME_PATH="/runpod-volume"

# Install Python 3.10 and necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
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
    && rm -rf /usr/local/cuda/targets/x86_64-linux/lib/*_static.a

# Set python3 alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Upgrade pip & install PyTorch with CUDA 12.1 (using buildkit pip cache mount)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Copy dependency definitions and install Python packages
COPY requirements-runpod.txt /app/requirements-runpod.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r /app/requirements-runpod.txt

# Build native custom rasterizer and differentiable renderer
COPY 3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/custom_rasterizer /tmp/custom_rasterizer
RUN cd /tmp/custom_rasterizer && python3 setup.py install && rm -rf /tmp/custom_rasterizer

# Build beside the Python fallback for the package-relative import.
COPY 3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer/setup.py 3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer/mesh_processor.cpp /app/3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer/
RUN cd /app/3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer && python3 setup.py build_ext --inplace && rm -rf build && \
    python3 -c "import mesh_processor as m; assert m.__file__.endswith('.so'), m.__file__"

COPY 3d_generative/Hunyuan3D-2-main /app/3d_generative/Hunyuan3D-2-main

# Copy application handler, preload helper, and entrypoint
COPY preload_models.py /app/preload_models.py
COPY runpod_handler.py /app/runpod_handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run entrypoint script
CMD ["/app/start.sh"]
