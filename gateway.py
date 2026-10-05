import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import base64

NSFW_URL    = "http://127.0.0.1:5001/analyze"
CAPTION_URL = "http://127.0.0.1:5002/caption"
THREED_URL  = "http://127.0.0.1:5003/generate"
CUSTOM_URL  = "http://127.0.0.1:5004/custom_describe"

app = FastAPI(title="Gateway", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health(): return {"ok": True}

# ---------- NSFW ----------
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    files = {"image": (image.filename, data, image.content_type)}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(NSFW_URL, files=files)
        except Exception as e:
            raise HTTPException(502, f"NSFW service error: {e}")
    return JSONResponse(r.json(), status_code=r.status_code)

# ---------- Caption (ĐÃ UPDATE) ----------
@app.post("/caption")
async def caption(
    image: UploadFile = File(...),
    prompt: str | None = Form(None),
    max_new_tokens: int = Form(60),
    min_new_tokens: int = Form(20) # <--- Nhận từ HTML
):
    data = await image.read()
    files = {"image": (image.filename, data, image.content_type)}
    
    # Chuyển tiếp params sang main.py
    params = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens
    }
    if prompt:
        params["prompt"] = prompt

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(CAPTION_URL, files=files, params=params)
        except Exception as e:
            raise HTTPException(502, f"Caption service error: {e}")

    return JSONResponse(r.json(), status_code=r.status_code)

# ---------- 3D ----------
@app.post("/generate3d")
async def generate3d(image: UploadFile = File(...)):
    raw = await image.read()
    
    # 1. Chuyển sang Base64
    base64_str = base64.b64encode(raw).decode("utf-8")
    
    # 2. Đóng gói JSON
    payload = {
        "image": base64_str,
        "texture": True,  # <--- Thêm dòng này để yêu cầu vẽ màu
        "texture_resolution": 1024 # (Tùy chọn) Độ phân giải texture
    }

    async with httpx.AsyncClient(timeout=1200.0) as client: # Tăng timeout lên 1200s (20 phút)
        try:
            r = await client.post(THREED_URL, json=payload)
        except Exception as e:
            raise HTTPException(502, f"3D service error: {e}")
            
    if r.status_code != 200:
        print(f"Service 3D Error: {r.text}") 
        raise HTTPException(status_code=r.status_code, detail=r.text)
        
    # --- ĐOẠN SỬA LỖI Ở ĐÂY ---
    # Dùng Response thay vì StreamingResponse vì r.content đã là bytes hoàn chỉnh
    return Response(
        content=r.content, 
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'inline; filename="model.glb"'}
    )

# --- SERVICE MỚI (CHỈ THÊM ĐOẠN NÀY) ---
@app.post("/custom_describe")
async def custom_describe(image: UploadFile = File(...), prompt: str = Form("Mô tả ảnh này")):
    data = await image.read()
    files = {"image": (image.filename, data, image.content_type)}
    form_data = {"prompt": prompt}
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            r = await client.post(CUSTOM_URL, files=files, data=form_data)
        except Exception as e:
            raise HTTPException(502, f"Custom AI service error (Port 5004): {e}")
    return JSONResponse(r.json(), status_code=r.status_code)
