"""
Runpod Serverless Unified Worker Handler
Supports 4 actions:
1. analyze: NSFW detection (CPU)
2. caption: BLIP image captioning (CPU)
3. custom_describe: LLaVA 7B via Ollama (GPU with immediate unload)
4. generate3d: Hunyuan3D-2 geometry + texture generation (GPU with low-VRAM offloading)
"""

import base64
import gc
import hashlib
import io
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, Generator, Optional, Tuple

from PIL import Image
import httpx
import runpod
import torch

# Ensure hy3dgen in subfolder can be imported if not installed in site-packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HUNYUAN_DIR = os.path.join(CURRENT_DIR, "3dgen", "Hunyuan3D-2-main")
if HUNYUAN_DIR not in sys.path:
    sys.path.insert(0, HUNYUAN_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("runpod_handler")

# Constants & Configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llava:7b")
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-large")
NSFW_MODEL_NAME = os.getenv("NSFW_MODEL_NAME", "strangerguardhf/nsfw_image_detection")
HUNYUAN_SHAPE_MODEL = os.getenv("HUNYUAN_SHAPE_MODEL", "tencent/Hunyuan3D-2mini")
HUNYUAN_SHAPE_SUBFOLDER = os.getenv("HUNYUAN_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
HUNYUAN_TEX_MODEL = os.getenv("HUNYUAN_TEX_MODEL", "tencent/Hunyuan3D-2")

# Chunk size for GLB streaming: 512 KiB raw data per chunk
CHUNK_SIZE_BYTES = 512 * 1024
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max payload

NSFW_LABELS = [
    "Anime Picture",
    "Hentai",
    "Normal",
    "Pornography",
    "Enticing or Sensual",
]
NSFW_BAD_LABELS = {"Pornography", "Hentai", "Enticing or Sensual"}


def get_vram_info() -> Optional[Dict[str, float]]:
    """Return current VRAM usage in MB if CUDA is available."""
    if torch.cuda.is_available():
        allocated = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        reserved = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        max_allocated = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
        }
    return None


def clean_vram():
    """Trigger Python GC and empty PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_base64_image(raw_str: str) -> Tuple[Image.Image, bytes]:
    """
    Decodes base64 string (with or without data URI header) to PIL Image and raw bytes.
    Validates payload format and image integrity.
    """
    if not raw_str or not isinstance(raw_str, str):
        raise ValueError("Image data is missing or not a valid string.")

    # Remove data URI header if present (e.g. 'data:image/png;base64,...')
    if "," in raw_str and ";base64" in raw_str:
        raw_str = raw_str.split(",", 1)[1]

    raw_str = raw_str.strip()
    try:
        image_bytes = base64.b64decode(raw_str)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {str(e)}")

    if len(image_bytes) == 0:
        raise ValueError("Decoded image data is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image size exceeds maximum allowed limit ({len(image_bytes)} > {MAX_IMAGE_SIZE_BYTES} bytes)."
        )

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # verify image structure
        # Reopen after verify
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image format or corrupted image: {str(e)}")

    return pil_img, image_bytes


def chunk_bytes(data: bytes, chunk_size: int = CHUNK_SIZE_BYTES) -> list:
    """Split bytes into chunks of specified size."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


class ModelManager:
    """Singleton-style manager for lazy loading and lifecycle management of ML models."""

    _instance = None

    def __init__(self):
        self.nsfw_processor = None
        self.nsfw_model = None
        self.blip_processor = None
        self.blip_model = None
        self.hunyuan_worker = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_nsfw(self):
        """Lazy load NSFW classification model on CPU."""
        if self.nsfw_model is None:
            logger.info(f"Loading NSFW model ({NSFW_MODEL_NAME}) on CPU...")
            from transformers import AutoImageProcessor, SiglipForImageClassification

            self.nsfw_processor = AutoImageProcessor.from_pretrained(NSFW_MODEL_NAME)
            self.nsfw_model = SiglipForImageClassification.from_pretrained(
                NSFW_MODEL_NAME
            ).to("cpu")
            self.nsfw_model.eval()
            logger.info("NSFW model loaded successfully on CPU.")
        return self.nsfw_processor, self.nsfw_model

    def load_blip(self):
        """Lazy load BLIP caption model on CPU."""
        if self.blip_model is None:
            logger.info(f"Loading BLIP model ({BLIP_MODEL_NAME}) on CPU...")
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_NAME
            ).to("cpu")
            self.blip_model.eval()
            logger.info("BLIP model loaded successfully on CPU.")
        return self.blip_processor, self.blip_model

    def load_hunyuan(self):
        """Lazy load Hunyuan3D pipeline with CPU offloading."""
        if self.hunyuan_worker is None:
            logger.info("Loading Hunyuan3D pipelines...")
            from hy3dgen.rembg import BackgroundRemover
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            rembg = BackgroundRemover()

            logger.info(
                f"Loading shape model {HUNYUAN_SHAPE_MODEL} (subfolder: {HUNYUAN_SHAPE_SUBFOLDER})..."
            )
            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN_SHAPE_MODEL,
                subfolder=HUNYUAN_SHAPE_SUBFOLDER,
                use_safetensors=True,
                device="cpu",
            )
            try:
                shape_pipeline.enable_flashvdm(mc_algo="mc")
            except Exception as e:
                logger.warning(f"Could not enable flashvdm: {e}")

            logger.info(f"Loading texture model {HUNYUAN_TEX_MODEL}...")
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_TEX_MODEL)

            # Offload texture pipeline parts to CPU
            if hasattr(tex_pipeline, "unet") and tex_pipeline.unet is not None:
                tex_pipeline.unet.to("cpu")
            if hasattr(tex_pipeline, "vae") and tex_pipeline.vae is not None:
                tex_pipeline.vae.to("cpu")
            if hasattr(tex_pipeline, "text_encoder") and tex_pipeline.text_encoder is not None:
                tex_pipeline.text_encoder.to("cpu")

            clean_vram()
            self.hunyuan_worker = {
                "rembg": rembg,
                "shape_pipeline": shape_pipeline,
                "tex_pipeline": tex_pipeline,
            }
            logger.info("Hunyuan3D pipelines initialized and offloaded to CPU.")
        return self.hunyuan_worker

    @staticmethod
    def unload_ollama(model_name: str = OLLAMA_MODEL_NAME):
        """Force Ollama daemon to immediately unload the model from GPU VRAM."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.post(
                    f"{OLLAMA_API_URL}/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": 0},
                )
                logger.info(f"Ollama unload response status: {res.status_code}")
        except Exception as e:
            logger.warning(f"Could not contact Ollama daemon to unload model: {e}")
        clean_vram()


# Initialize singleton manager
model_manager = ModelManager.get_instance()


def handle_analyze(
    job_input: Dict[str, Any], start_time: float
) -> Generator[Dict[str, Any], None, None]:
    """Execute NSFW classification on CPU."""
    yield {
        "type": "progress",
        "stage": "decode_input",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    img_str = job_input.get("image_base64") or job_input.get("image")
    pil_img, _ = decode_base64_image(img_str)

    yield {
        "type": "progress",
        "stage": "worker_init",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    processor, model = model_manager.load_nsfw()

    yield {
        "type": "progress",
        "stage": "nsfw_inference",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    t0 = time.time()
    with torch.no_grad():
        inputs = processor(images=pil_img, return_tensors="pt").to("cpu")
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

    inference_ms = int((time.time() - t0) * 1000)
    scores = {
        NSFW_LABELS[i]: round(float(probs[i]), 4)
        for i in range(min(len(NSFW_LABELS), len(probs)))
    }
    top_label = max(scores, key=scores.get)
    is_nsfw = (
        scores.get("Pornography", 0) >= 0.50
        or scores.get("Hentai", 0) >= 0.50
        or scores.get("Enticing or Sensual", 0) >= 0.70
        or top_label in NSFW_BAD_LABELS
    )

    total_ms = int((time.time() - start_time) * 1000)
    yield {
        "type": "result",
        "action": "analyze",
        "data": {
            "ok": True,
            "scores": scores,
            "top_label": top_label,
            "is_nsfw": is_nsfw,
        },
        "timings": {
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    }


def handle_caption(
    job_input: Dict[str, Any], start_time: float
) -> Generator[Dict[str, Any], None, None]:
    """Execute BLIP image captioning on CPU."""
    yield {
        "type": "progress",
        "stage": "decode_input",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    img_str = job_input.get("image_base64") or job_input.get("image")
    pil_img, _ = decode_base64_image(img_str)
    prompt = job_input.get("prompt")
    max_tokens = int(job_input.get("max_new_tokens", 60))
    min_tokens = int(job_input.get("min_new_tokens", 20))

    yield {
        "type": "progress",
        "stage": "worker_init",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    processor, model = model_manager.load_blip()

    yield {
        "type": "progress",
        "stage": "caption_inference",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    t0 = time.time()
    with torch.no_grad():
        if prompt:
            inputs = processor(pil_img, text=prompt, return_tensors="pt").to("cpu")
        else:
            inputs = processor(pil_img, return_tensors="pt").to("cpu")

        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            num_beams=3,
        )
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

    inference_ms = int((time.time() - t0) * 1000)
    total_ms = int((time.time() - start_time) * 1000)

    yield {
        "type": "result",
        "action": "caption",
        "data": {
            "ok": True,
            "caption": caption,
        },
        "timings": {
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    }


def handle_custom_describe(
    job_input: Dict[str, Any], start_time: float
) -> Generator[Dict[str, Any], None, None]:
    """Execute Ollama LLaVA 7B custom description with GPU unload."""
    yield {
        "type": "progress",
        "stage": "decode_input",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    img_str = job_input.get("image_base64") or job_input.get("image")
    _, raw_img_bytes = decode_base64_image(img_str)
    b64_clean = base64.b64encode(raw_img_bytes).decode("utf-8")

    user_prompt = job_input.get("prompt", "Mô tả ảnh này")
    model_name = job_input.get("model", OLLAMA_MODEL_NAME)

    yield {
        "type": "progress",
        "stage": "preprocess",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    # Translate Prompt to English if needed
    translated_prompt = user_prompt
    try:
        from deep_translator import GoogleTranslator

        translated_prompt = GoogleTranslator(source="auto", target="en").translate(
            user_prompt
        )
    except Exception as e:
        logger.warning(f"Translation to English failed, using original prompt: {e}")

    full_prompt = f"Describe this image in detail and fulfill this request: {translated_prompt}"

    yield {
        "type": "progress",
        "stage": "llava_inference",
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "vram": get_vram_info(),
    }

    t0 = time.time()
    try:
        with httpx.Client(timeout=180.0) as client:
            res = client.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "images": [b64_clean],
                    "stream": False,
                },
            )
            if res.status_code != 200:
                raise RuntimeError(
                    f"Ollama error (HTTP {res.status_code}): {res.text}"
                )
            english_result = res.json().get("response", "").strip()
    finally:
        # Guarantee unload
        model_manager.unload_ollama(model_name)

    inference_ms = int((time.time() - t0) * 1000)

    # Translate result back to Vietnamese if requested or preserve English
    vietnamese_result = english_result
    try:
        from deep_translator import GoogleTranslator

        vietnamese_result = GoogleTranslator(source="en", target="vi").translate(
            english_result
        )
    except Exception as e:
        logger.warning(f"Translation back to Vietnamese failed: {e}")

    total_ms = int((time.time() - start_time) * 1000)

    yield {
        "type": "result",
        "action": "custom_describe",
        "data": {
            "ok": True,
            "description": vietnamese_result,
            "description_en": english_result,
        },
        "timings": {
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    }


def handle_generate3d(
    job_input: Dict[str, Any], start_time: float
) -> Generator[Dict[str, Any], None, None]:
    """Execute Hunyuan3D-2 3D mesh + texture generation with stream chunking."""
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    yield {
        "type": "progress",
        "stage": "decode_input",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }

    img_str = job_input.get("image_base64") or job_input.get("image")
    pil_img, _ = decode_base64_image(img_str)

    seed = int(job_input.get("seed", 1234))
    octree_res = int(job_input.get("octree_resolution", 256))
    steps = int(job_input.get("num_inference_steps", 5))
    guidance = float(job_input.get("guidance_scale", 5.0))
    face_count = int(job_input.get("face_count", 40000))
    enable_texture = bool(job_input.get("texture", True))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    yield {
        "type": "progress",
        "stage": "worker_init",
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "vram": get_vram_info(),
    }

    hunyuan = model_manager.load_hunyuan()
    rembg = hunyuan["rembg"]
    shape_pipeline = hunyuan["shape_pipeline"]
    tex_pipeline = hunyuan["tex_pipeline"]

    # 1. Preprocess & background removal
    yield {
        "type": "progress",
        "stage": "preprocess",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }
    t_prep_start = time.time()
    image_no_bg = rembg(pil_img)
    prep_ms = int((time.time() - t_prep_start) * 1000)

    # 2. Shape generation
    yield {
        "type": "progress",
        "stage": "shape_generation",
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "vram": get_vram_info(),
    }
    t_shape_start = time.time()

    logger.info(f"Offloading shape pipeline to {device}...")
    shape_pipeline.to(device)

    try:
        generator = torch.Generator("cpu").manual_seed(seed)
        mesh = shape_pipeline(
            image=image_no_bg,
            generator=generator,
            octree_resolution=octree_res,
            num_inference_steps=steps,
            guidance_scale=guidance,
            mc_algo="mc",
        )[0]
    finally:
        logger.info("Offloading shape pipeline back to CPU...")
        shape_pipeline.to("cpu")
        clean_vram()

    shape_ms = int((time.time() - t_shape_start) * 1000)

    if mesh is None:
        raise RuntimeError("Hunyuan3D shape generation returned empty mesh.")

    # 3. Mesh cleanup and reduction
    yield {
        "type": "progress",
        "stage": "mesh_cleanup",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }
    t_clean_start = time.time()
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=face_count)
    clean_ms = int((time.time() - t_clean_start) * 1000)

    # 4. Texture generation
    tex_ms = 0
    if enable_texture:
        yield {
            "type": "progress",
            "stage": "texture_generation",
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "vram": get_vram_info(),
        }
        t_tex_start = time.time()
        try:
            if hasattr(tex_pipeline, "unet") and tex_pipeline.unet is not None:
                tex_pipeline.unet.to(device)
            if hasattr(tex_pipeline, "vae") and tex_pipeline.vae is not None:
                tex_pipeline.vae.to(device)
            if (
                hasattr(tex_pipeline, "text_encoder")
                and tex_pipeline.text_encoder is not None
            ):
                tex_pipeline.text_encoder.to(device)

            mesh = tex_pipeline(mesh, image_no_bg)
        except Exception as e:
            logger.warning(
                f"Texture generation encountered an error: {e}. Falling back to untextured mesh."
            )
        finally:
            if hasattr(tex_pipeline, "unet") and tex_pipeline.unet is not None:
                tex_pipeline.unet.to("cpu")
            if hasattr(tex_pipeline, "vae") and tex_pipeline.vae is not None:
                tex_pipeline.vae.to("cpu")
            if (
                hasattr(tex_pipeline, "text_encoder")
                and tex_pipeline.text_encoder is not None
            ):
                tex_pipeline.text_encoder.to("cpu")
            clean_vram()

        tex_ms = int((time.time() - t_tex_start) * 1000)

    # 5. GLB Export
    yield {
        "type": "progress",
        "stage": "glb_export",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }
    t_exp_start = time.time()
    glb_buffer = io.BytesIO()
    mesh.export(glb_buffer, file_type="glb")
    glb_bytes = glb_buffer.getvalue()
    export_ms = int((time.time() - t_exp_start) * 1000)

    # 6. Chunking & Streaming
    sha256_hash = hashlib.sha256(glb_bytes).hexdigest()
    chunks = chunk_bytes(glb_bytes, CHUNK_SIZE_BYTES)
    total_chunks = len(chunks)
    total_size = len(glb_bytes)

    yield {
        "type": "file_meta",
        "name": "model.glb",
        "size": total_size,
        "chunks": total_chunks,
        "sha256": sha256_hash,
    }

    for idx, chunk in enumerate(chunks):
        b64_chunk = base64.b64encode(chunk).decode("utf-8")
        yield {
            "type": "file_chunk",
            "index": idx,
            "data": b64_chunk,
        }

    total_ms = int((time.time() - start_time) * 1000)
    yield {
        "type": "done",
        "timings": {
            "preprocess_ms": prep_ms,
            "shape_ms": shape_ms,
            "cleanup_ms": clean_ms,
            "texture_ms": tex_ms,
            "export_ms": export_ms,
            "total_ms": total_ms,
        },
        "vram": get_vram_info(),
    }


def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Main entrypoint generator for Runpod Serverless.
    Parses request input, routes to action handler, and yields stream events.
    """
    start_time = time.time()
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        yield {
            "type": "error",
            "stage": "validate_input",
            "error": "Request 'input' must be a JSON object.",
            "timings": {"total_ms": 0},
        }
        return

    action = job_input.get("action")
    if not action:
        yield {
            "type": "error",
            "stage": "validate_input",
            "error": "Missing required field 'action' in input. Expected: analyze, caption, custom_describe, generate3d.",
            "timings": {"total_ms": 0},
        }
        return

    logger.info(f"Starting job with action='{action}'")

    try:
        if action == "analyze":
            yield from handle_analyze(job_input, start_time)
        elif action == "caption":
            yield from handle_caption(job_input, start_time)
        elif action == "custom_describe":
            yield from handle_custom_describe(job_input, start_time)
        elif action == "generate3d":
            yield from handle_generate3d(job_input, start_time)
        else:
            yield {
                "type": "error",
                "stage": "validate_input",
                "error": f"Invalid action '{action}'. Supported actions: analyze, caption, custom_describe, generate3d.",
                "timings": {"total_ms": int((time.time() - start_time) * 1000)},
            }
    except Exception as e:
        logger.error(f"Error processing action '{action}': {traceback.format_exc()}")
        yield {
            "type": "error",
            "stage": "execution",
            "error": str(e),
            "timings": {"total_ms": int((time.time() - start_time) * 1000)},
        }
    finally:
        clean_vram()


if __name__ == "__main__":
    logger.info("Starting Runpod Serverless Worker...")
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
