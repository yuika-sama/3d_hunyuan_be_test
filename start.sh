#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Starting Runpod Serverless Worker Container"
echo "=================================================="

# 1. Start Ollama daemon in background
echo "[STARTUP] Starting Ollama daemon..."
ollama serve &
OLLAMA_PID=$!

# 2. Wait for Ollama daemon health check
echo "[STARTUP] Waiting for Ollama daemon on http://127.0.0.1:11434..."
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "[STARTUP] Ollama daemon is healthy and ready!"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "[WARNING] Ollama daemon health check timed out. Proceeding anyway..."
        break
    fi
    sleep 1
done

# 3. Start Runpod Serverless Python Worker
echo "[STARTUP] Starting Runpod Serverless Python handler..."
exec python3 -u /app/runpod_handler.py
