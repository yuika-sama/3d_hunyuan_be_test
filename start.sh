#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Starting Runpod Serverless Worker Container"
echo "=================================================="

# Export Hugging Face download configuration
export HF_HUB_DISABLE_XET="1"
export HF_HUB_ENABLE_HF_TRANSFER="0"
export HF_HUB_DOWNLOAD_TIMEOUT="600"

# 1. Preload models into local cache before announcing worker ready to Runpod
if [ "${PRELOAD_MODELS:-1}" = "1" ]; then
    echo "[STARTUP] Pre-downloading / verifying Hunyuan3D models before accepting jobs..."
    python3 /app/preload_models.py --full || echo "[STARTUP WARNING] Model preloading encountered an issue, proceeding to start worker handler..."
fi

# 2. Start Runpod Serverless Python Worker
echo "[STARTUP] Starting Runpod Serverless Python handler..."
exec python3 -u /app/runpod_handler.py
