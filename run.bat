@echo off
echo ====== STARTING ALL SERVERS ======

if not defined GATEWAY_API_KEY (
    echo Set GATEWAY_API_KEY before starting the gateway.
    exit /b 1
)

REM --- Start GATEWAY (Main Entry Port 4000) ---
start cmd /k "uvicorn gateway:app --host 127.0.0.1 --port 4000"

REM --- Start NSFW AI (Port 5001) ---
start cmd /k "cd nsfw-image-detect && uvicorn service:app --host 127.0.0.1 --port 5001"

REM --- Start Hunyuan3D (Port 5003) ---
start cmd /k "cd 3d_generative/Hunyuan3D-2-main && python api_server.py --host 127.0.0.1 --port 5003 --enable_tex"

echo All servers launched!
pause
