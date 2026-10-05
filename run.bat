@echo off
echo ====== STARTING ALL SERVERS ======

REM --- Start NGROK ---
start cmd /k "ngrok http 4000"

REM --- Start GATEWAY (Main Entry Port 4000) ---
start cmd /k "uvicorn gateway:app --host 0.0.0.0 --port 4000"

REM --- Start NSFW AI (Port 5001) ---
start cmd /k "cd nsfw-image-detect && uvicorn service:app --host 0.0.0.0 --port 5001"

REM --- Start BLIP Captioning (Port 5002) ---
start cmd /k "cd blip-image-captioning-api-main && uvicorn app.main:app --host 0.0.0.0 --port 5002"

REM --- Start Hunyuan3D (Port 5003) ---
start cmd /k "cd 3dgen/Hunyuan3D-2-main && python api_server_backup.py --host 0.0.0.0 --port 5003 --enable_tex"

REM --- Start Custom Ollama Service (Port 5004) ---
start cmd /k "cd advance-image-to-text && python ollama_service.py"

echo All servers launched!
pause