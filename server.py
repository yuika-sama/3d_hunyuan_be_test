# server.py  — ONE-PORT API (NSFW + CAPTION)
import io
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
NSFW_DIR = BASE_DIR / "nsfw-image-detect"
CAPTION_DIR = BASE_DIR / "blip-image-captioning-api-main"

SAVE_UPLOADS = os.getenv("SAVE_UPLOADS", "0") == "1"
UPLOAD_DIR = BASE_DIR / "uploads"
if SAVE_UPLOADS:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# BLIP model config (override qua env nếu muốn)
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-large")
BLIP_MAX_NEW_TOKENS = int(os.getenv("BLIP_MAX_NEW_TOKENS", "60"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# App
# -------------------------
app = FastAPI(title="Unified AI API (NSFW + Caption)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # chỉnh lại nếu cần
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load NSFW model + labels
# -------------------------
print("[INIT] Loading NSFW model...")
nsfw_processor = AutoImageProcessor.from_pretrained(
    "strangerguardhf/nsfw_image_detection"
)
nsfw_model = SiglipForImageClassification.from_pretrained(
    "strangerguardhf/nsfw_image_detection"
).to(DEVICE)

# labels.txt nằm trong nsfw-image-detect/
labels_path = NSFW_DIR / "labels.txt"
if labels_path.exists():
    with open(labels_path, "r", encoding="utf-8") as f:
        NSFW_LABELS = [ln.strip() for ln in f if ln.strip()]
else:
    # fallback nếu thiếu file
    NSFW_LABELS = [
        "Anime Picture",
        "Hentai",
        "Normal",
        "Pornography",
        "Enticing or Sensual",
    ]
NSFW_BAD = {"Pornography", "Hentai", "Enticing or Sensual"}  # flag là nsfw

# -------------------------
# Load BLIP (caption)
# -------------------------
print("[INIT] Loading BLIP caption model...")
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)
blip_model.eval()

# -------------------------
# Utils
# -------------------------
def _read_pil(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _maybe_save(image_bytes: bytes, orig_name: str) -> Optional[str]:
    if not SAVE_UPLOADS:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = orig_name.replace("/", "_").replace("\\", "_")
    out_path = UPLOAD_DIR / f"{ts}_{safe_name}"
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return str(out_path)

# -------------------------
# Schemas
# -------------------------
class HealthResp(BaseModel):
    ok: bool = True

class NSFWResp(BaseModel):
    ok: bool
    scores: Dict[str, float]
    top_label: str
    is_nsfw: bool
    saved_path: Optional[str] = None

class CaptionResp(BaseModel):
    ok: bool
    caption: str
    saved_path: Optional[str] = None

# -------------------------
# Routes
# -------------------------
@app.get("/health", response_model=HealthResp)
def health():
    return {"ok": True}

@app.post("/analyze", response_model=NSFWResp)
async def analyze(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await image.read()

    saved = _maybe_save(data, image.filename)
    pil = _read_pil(data)

    with torch.no_grad():
        inputs = nsfw_processor(images=pil, return_tensors="pt").to(DEVICE)
        logits = nsfw_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    scores = {NSFW_LABELS[i]: float(probs[i]) for i in range(len(NSFW_LABELS))}
    top_idx = int(torch.tensor(probs).argmax().item())
    top_label = NSFW_LABELS[top_idx]
    is_nsfw = top_label in NSFW_BAD

    return {
        "ok": True,
        "scores": scores,
        "top_label": top_label,
        "is_nsfw": is_nsfw,
        "saved_path": saved,
    }

@app.post("/caption", response_model=CaptionResp)
async def caption(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    data = await image.read()
    saved = _maybe_save(data, image.filename)
    pil = _read_pil(data)

    with torch.no_grad():
        inputs = blip_processor(pil, text=prompt, return_tensors="pt").to(DEVICE)
        out = blip_model.generate(
            **inputs,
            max_new_tokens=BLIP_MAX_NEW_TOKENS,
            num_beams=3,
        )
        cap = blip_processor.decode(out[0], skip_special_tokens=True).strip()

    return {"ok": True, "caption": cap, "saved_path": saved}
