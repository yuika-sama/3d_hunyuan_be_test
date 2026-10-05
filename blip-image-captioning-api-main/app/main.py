# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import logging, logging.config
from pathlib import Path
from typing import Optional

# Dùng thư viện dịch deep-translator
from deep_translator import GoogleTranslator

from .model import load_model, generate_caption
from .utils import load_image_from_file
from .config import settings

app = FastAPI(title="BLIP Image Captioning API")

# ----- Logging -----
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
LOG_PATH = APP_DIR / ".cache" / "app.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH_STR = LOG_PATH.as_posix()
CONF_PATH = PROJECT_ROOT / "logging.conf"
logging.config.fileConfig(
    fname=str(CONF_PATH), disable_existing_loggers=False,
    defaults={"logfile": LOG_PATH_STR}, encoding="utf-8",
)
logger = logging.getLogger(__name__)

# ----- Model & Translator -----
model = None
processor = None
translator = GoogleTranslator(source='auto', target='vi')

@app.on_event("startup")
async def _load_model_on_startup():
    global model, processor
    model, processor = load_model(settings.blip_model_name)
    logger.info("BLIP Model loaded at startup")

@app.post("/caption")
async def caption(
    image: UploadFile = File(...), 
    prompt: Optional[str] = Query(None),       # Nhận prompt từ Query Param
    max_new_tokens: int = Query(60)            # Nhận số lượng từ, mặc định 60
):
    try:
        if image.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=400, detail="Invalid image format (use jpg/png)")

        img_content = await image.read()
        img = load_image_from_file(img_content)
        
        # Gọi model với các tham số động từ người dùng
        raw_caption_en = generate_caption(
            model=model,
            processor=processor,
            image=img,
            prompt=prompt,
            max_new_tokens=max_new_tokens, # <-- QUAN TRỌNG: Dùng biến này thay vì settings
        )

        # Dịch sang tiếng Việt
        try:
            final_caption = translator.translate(raw_caption_en)
        except Exception:
            final_caption = raw_caption_en

        return {
            "caption": final_caption,
            "original_caption": raw_caption_en,
            "params_used": { # Trả về để debug xem server nhận được gì
                "prompt": prompt,
                "max_tokens": max_new_tokens
            }
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error inside /caption")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await image.close()