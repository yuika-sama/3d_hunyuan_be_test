import base64
import hmac
import io
import os
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

NSFW_URL = "http://127.0.0.1:5001/analyze"
THREED_URL = "http://127.0.0.1:5003/generate"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216


async def require_api_key(request: Request):
    if request.url.path == "/health":
        return
    key = os.getenv("GATEWAY_API_KEY")
    if not key:
        raise HTTPException(503, "Set GATEWAY_API_KEY before using the gateway")
    authorization = request.headers.get("authorization", "")
    if not hmac.compare_digest(authorization.encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Invalid gateway credentials")


app = FastAPI(title="Gateway", version="1.4", dependencies=[Depends(require_api_key)])


@app.get("/health")
async def health():
    return {"ok": True}


async def read_image(image: UploadFile):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Image must be JPEG or PNG")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(400, "Image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 6 MiB limit")
    try:
        with Image.open(io.BytesIO(data)) as decoded:
            if decoded.format not in {"JPEG", "PNG"}:
                raise ValueError("Unsupported image format")
            if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit")
            decoded.verify()
    except Exception as exc:
        raise HTTPException(400, "Invalid image or image exceeds 16 megapixels") from exc
    return data


async def _post_json(client: httpx.AsyncClient, url: str, service: str, **kwargs):
    try:
        response = await client.post(url, **kwargs)
        response.raise_for_status()
        result = response.json()
        if (not isinstance(result, dict) or result.get("ok") is not True
                or type(result.get("is_nsfw")) is not bool):
            raise ValueError("Invalid moderation response")
        return result
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"{service} service returned an invalid response") from exc


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await read_image(image)
    async with httpx.AsyncClient(timeout=60.0) as client:
        result = await _post_json(client, NSFW_URL, "NSFW", files={
            "image": (image.filename or "image.png", data, image.content_type),
        })
    return JSONResponse(result)


async def generate_checked(image, texture_image):
    """Both public generation routes use the same validation and moderation."""
    image_bytes = await read_image(image)
    has_texture = texture_image is not None
    texture_bytes = await read_image(texture_image) if has_texture else image_bytes
    async with httpx.AsyncClient(timeout=1200.0) as client:
        results = []
        sources = [(image, image_bytes, "moderation")]
        if has_texture:
            sources.append((texture_image, texture_bytes, "texture_moderation"))
        for upload, data, stage in sources:
            result = await _post_json(client, NSFW_URL, "NSFW", files={
                "image": (upload.filename or "image.png", data, upload.content_type),
            })
            if result["is_nsfw"]:
                raise HTTPException(400, {"stage": stage, "error": "Image violates NSFW policy", "details": result})
            results.append(result)
        try:
            response = await client.post(THREED_URL, json={
                "image": base64.b64encode(image_bytes).decode(),
                "texture_image": base64.b64encode(texture_bytes).decode(),
                "texture": True,
                "texture_resolution": 2048,
                "bake_exp": 4,
            })
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(502, "3D service failed") from exc
    glb = response.content
    if (len(glb) < 12 or glb[:4] != b"glTF" or int.from_bytes(glb[4:8], "little") != 2
            or int.from_bytes(glb[8:12], "little") != len(glb)):
        raise HTTPException(502, "3D service returned an invalid GLB")
    return glb, results[0], results[-1]


@app.post("/generate3d")
async def generate3d(image: UploadFile = File(...), texture_image: UploadFile | None = None):
    glb, _, _ = await generate_checked(image, texture_image)
    return Response(content=glb, media_type="model/gltf-binary",
                    headers={"Content-Disposition": 'inline; filename="model.glb"'})


@app.post("/pipeline/process-all")
async def process_all_pipeline(image: UploadFile = File(...), texture_image: UploadFile | None = None):
    glb, moderation, texture_moderation = await generate_checked(image, texture_image)
    task_id = uuid4()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{task_id}.glb").write_bytes(glb)
    return {
        "ok": True,
        "task_id": str(task_id),
        "results": {
            "moderation": moderation,
            "texture_moderation": texture_moderation,
            "model_3d": {"status": "success", "glb_url": f"/pipeline/result/{task_id}.glb"},
        },
    }


@app.get("/pipeline/result/{task_id}.glb")
async def download_pipeline_result(task_id: UUID):
    model_path = OUTPUT_DIR / f"{task_id}.glb"
    if not model_path.is_file():
        raise HTTPException(404, "3D model not found")
    return FileResponse(model_path, media_type="model/gltf-binary", filename=f"{task_id}.glb")
