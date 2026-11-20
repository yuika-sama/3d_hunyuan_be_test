#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Starting Runpod Serverless Worker Container"
echo "=================================================="

# 1. Volume & Cache directory configuration
# Runpod Serverless mounts Global Volumes and Network Volumes to /runpod-volume by default.
# Regular GPU Cloud Pods mount volumes to /workspace.
VOLUME_PATH="${RUNPOD_VOLUME_PATH:-}"
if [ -z "$VOLUME_PATH" ]; then
    if [ -d "/runpod-volume" ]; then
        VOLUME_PATH="/runpod-volume"
    elif [ -d "/workspace" ] && [ -w "/workspace" ]; then
        VOLUME_PATH="/workspace"
    fi
fi

if [ -n "$VOLUME_PATH" ] && [ -d "$VOLUME_PATH" ]; then
    echo "[STARTUP] RunPod Volume detected at: $VOLUME_PATH"

    # Direct cache directories to Volume if not explicitly overridden
    export HF_HOME="${HF_HOME:-$VOLUME_PATH/huggingface}"
    export TORCH_HOME="${TORCH_HOME:-$VOLUME_PATH/cache/torch}"
    export U2NET_HOME="${U2NET_HOME:-$VOLUME_PATH/cache/u2net}"
    export HY3DGEN_MODELS="${HY3DGEN_MODELS:-$VOLUME_PATH/models}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VOLUME_PATH/cache/triton}"

    mkdir -p "$HF_HOME" "$TORCH_HOME" "$U2NET_HOME" "$HY3DGEN_MODELS" "$TRITON_CACHE_DIR"

    echo "[STARTUP] Persistent cache paths active:"
    echo "  - HF_HOME:          $HF_HOME"
    echo "  - TORCH_HOME:       $TORCH_HOME"
    echo "  - U2NET_HOME:       $U2NET_HOME"
    echo "  - HY3DGEN_MODELS:   $HY3DGEN_MODELS"
else
    echo "[STARTUP] WARNING: RunPod Volume not detected (checked /runpod-volume, /workspace)."
    echo "[STARTUP] Falling back to ephemeral Container Disk for caching."
    export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
    export TORCH_HOME="${TORCH_HOME:-/root/.cache/torch}"
    export U2NET_HOME="${U2NET_HOME:-/root/.u2net}"
    export HY3DGEN_MODELS="${HY3DGEN_MODELS:-/root/.cache/hy3dgen}"
fi

# Diffusers copies executable Python modules with file metadata; use a POSIX filesystem.
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/bookaria-hf-modules}"

# Export Hugging Face download configuration
# IMPORTANT for RunPod Global Volumes (S3/FUSE): Disable symlinks to prevent broken references and chmod errors
export HF_HUB_DISABLE_XET="1"
export HF_HUB_ENABLE_HF_TRANSFER="0"
export HF_HUB_DOWNLOAD_TIMEOUT="600"
export HF_HUB_DISABLE_SYMLINKS="1"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

# Propagate HF_TOKEN to HUGGING_FACE_HUB_TOKEN if set
if [ -n "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# 2. Preload / verify models before announcing worker ready to Runpod
if [ "${PRELOAD_MODELS:-1}" = "1" ]; then
    echo "[STARTUP] Verifying / pre-downloading models to cache before accepting jobs..."
    if ! python3 /app/preload_models.py --full --nsfw; then
        echo "[STARTUP WARNING] Model preload reported warnings or errors."
        if [ "${PRELOAD_FAIL_FATAL:-0}" = "1" ]; then
            echo "[STARTUP ERROR] PRELOAD_FAIL_FATAL=1 is set. Exiting container."
            exit 1
        fi
        echo "[STARTUP WARNING] Continuing to start handler (missing models will be fetched or handled gracefully on-demand)..."
    fi
fi

# 3. Start Runpod Serverless Python Worker
echo "[STARTUP] Starting Runpod Serverless Python handler..."
exec python3 -u /app/runpod_handler.py
