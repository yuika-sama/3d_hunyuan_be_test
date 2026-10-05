import os
import base64
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from api_server import ModelWorker  # dùng lại ModelWorker trong api_server.py

# Thư mục output – GLB (có texture) sẽ được copy về đây
OUTPUT_DIR = r"C:\Users\TomaszPC\Desktop\CODING\output"

app = FastAPI(
    title="Simple Hunyuan3D Image2Mesh Server (with Texture)",
    version="1.1",
)

# CORS cho phép gọi từ file HTML local / ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

worker: Optional[ModelWorker] = None


def init_worker():
    """
    Load model Hunyuan3D vào GPU.
    BẬT TEXTURE: enable_tex=True + tex_model_path = bản full.
    """
    global worker
    if worker is not None:
        return

    # NOTE:
    # - model_path: mini (shape)
    # - tex_model_path: full (texture)
    # - subfolder: đúng theo repo mày đang dùng
    # - device: "cuda" (bắt buộc có GPU nếu bật texture, CPU gần như không chịu nổi)
    worker = ModelWorker(
        model_path="tencent/Hunyuan3D-2mini",
        tex_model_path="tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        device="cuda",
        enable_tex=True,   # <-- BẬT TEXTURE Ở ĐÂY
    )


@app.on_event("startup")
async def on_startup():
    init_worker()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/generate3d")
async def generate3d(image: UploadFile = File(...)):
    """
    Nhận 1 ảnh, encode base64, gọi ModelWorker.generate với texture=True,
    trả về GLB đã bake texture.
    """
    if worker is None:
        init_worker()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file.")

    # Encode base64 giống api_server.py
    img_b64 = base64.b64encode(raw).decode("utf-8")

    # ---- PARAMS CHO TEXTURE MODE ----
    params = {
        "image": img_b64,
        "texture": True,          # <-- QUAN TRỌNG: yêu cầu gen texture
        "type": "glb",            # trả về dạng GLB
        "seed": 1234,
        "octree_resolution": 256, # có thể tăng 256 nếu GPU chịu được
        "num_inference_steps": 5, # tăng hơn 5 cho đẹp hơn (nhưng chậm hơn)
        "guidance_scale": 5.0,
    }

    try:
        uid = uuid.uuid4()

        # Tùy implementation ModelWorker, thường trả:
        #   file_path = đường dẫn GLB cuối cùng
        #   meta      = dict/info thêm
        file_path, _ = worker.generate(uid, params)

        if not os.path.isfile(file_path):
            raise RuntimeError(f"ModelWorker.generate không trả về file hợp lệ: {file_path}")

        # ---- COPY GLB SANG THƯ MỤC OUTPUT CỐ ĐỊNH ----
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{uid}.glb"
        target_path = os.path.join(OUTPUT_DIR, filename)

        import shutil
        shutil.copy(file_path, target_path)

        print("OK - GLB with texture saved to:", target_path)

    except Exception as e:
        # Vứt stacktrace ra log cho dễ debug
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    return FileResponse(
        path=target_path,
        media_type="model/gltf-binary",
        filename="model.glb",
    )


if __name__ == "__main__":
    uvicorn.run(
        "simple_image2mesh_server:app",
        host="0.0.0.0",
        port=5003,
        reload=False,
    )
