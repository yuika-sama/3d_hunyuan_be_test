import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import base64
from pathlib import Path
from uuid import UUID, uuid4

NSFW_URL    = "http://127.0.0.1:5001/analyze"
CAPTION_URL = "http://127.0.0.1:5002/caption"
THREED_URL  = "http://127.0.0.1:5003/generate"
CUSTOM_URL  = "http://127.0.0.1:5004/custom_describe"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

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


async def _post_json(client: httpx.AsyncClient, url: str, service: str, **kwargs):
    try:
        response = await client.post(url, **kwargs)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"{service} service error: {exc}") from exc

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
async def generate3d(
    image: UploadFile = File(...),
    texture_image: UploadFile | None = None,
    texture_resolution: int = Form(2048),
    bake_exp: int = Form(4),
    face_count: int = Form(45000),
    octree_resolution: int = Form(384),
):
    raw = await image.read()
    if texture_image is not None and hasattr(texture_image, "read"):
        texture_raw = await texture_image.read()
    else:
        texture_raw = raw
    
    # 1. Chuyển sang Base64
    base64_str = base64.b64encode(raw).decode("utf-8")
    texture_base64_str = base64.b64encode(texture_raw).decode("utf-8") if texture_raw else base64_str
    
    # 2. Đóng gói JSON
    payload = {
        "image": base64_str,
        "texture_image": texture_base64_str,
        "texture": True,
        "texture_resolution": texture_resolution,
        "bake_exp": bake_exp,
        "face_count": face_count,
        "octree_resolution": octree_resolution,
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


@app.post("/pipeline/process-all")
async def process_all_pipeline(
    image: UploadFile = File(...),
    texture_image: UploadFile | None = None,
    texture_resolution: int = Form(2048),
    bake_exp: int = Form(4),
):
    """Run safety checks, then textured 3D generation using a single image (or optional texture override)."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Image is empty")
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Image must be JPEG or PNG")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds the 20 MB limit")

    has_separate_texture = texture_image is not None and hasattr(texture_image, "read")
    if has_separate_texture:
        texture_image_bytes = await texture_image.read()
        if not texture_image_bytes:
            raise HTTPException(400, "Texture image is empty")
        if texture_image.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "Texture image must be JPEG or PNG")
        if len(texture_image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(413, "Texture image exceeds the 20 MB limit")
    else:
        texture_image_bytes = image_bytes

    files = {
        "image": (
            image.filename or "image.png",
            image_bytes,
            image.content_type or "application/octet-stream",
        )
    }

    async with httpx.AsyncClient(timeout=1200.0) as client:
        moderation = await _post_json(client, NSFW_URL, "NSFW", files=files)
        if moderation.get("is_nsfw", False):
            raise HTTPException(
                400,
                {"stage": "moderation", "error": "Image violates NSFW policy", "details": moderation},
            )

        if has_separate_texture:
            texture_files = {
                "image": (
                    texture_image.filename or "texture.png",
                    texture_image_bytes,
                    texture_image.content_type or "application/octet-stream",
                )
            }
            texture_moderation = await _post_json(client, NSFW_URL, "NSFW", files=texture_files)
            if texture_moderation.get("is_nsfw", False):
                raise HTTPException(
                    400,
                    {
                        "stage": "texture_moderation",
                        "error": "Texture image violates NSFW policy",
                        "details": texture_moderation,
                    },
                )
        else:
            texture_moderation = moderation

        try:
            model_response = await client.post(
                THREED_URL,
                json={
                    "image": base64.b64encode(image_bytes).decode("utf-8"),
                    "texture_image": base64.b64encode(texture_image_bytes).decode("utf-8"),
                    "texture": True,
                    "texture_resolution": texture_resolution,
                    "bake_exp": bake_exp,
                },
            )
            model_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"3D service error: {exc}") from exc

    task_id = uuid4()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{task_id}.glb").write_bytes(model_response.content)

    return {
        "ok": True,
        "task_id": str(task_id),
        "results": {
            "moderation": moderation,
            "texture_moderation": texture_moderation,
            "model_3d": {
                "status": "success",
                "glb_url": f"/pipeline/result/{task_id}.glb",
            },
        },
    }


@app.get("/pipeline/result/{task_id}.glb")
async def download_pipeline_result(task_id: UUID):
    model_path = OUTPUT_DIR / f"{task_id}.glb"
    if not model_path.is_file():
        raise HTTPException(404, "3D model not found")
    return FileResponse(model_path, media_type="model/gltf-binary", filename=f"{task_id}.glb")
