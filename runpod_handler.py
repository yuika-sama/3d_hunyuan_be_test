"""
Runpod Serverless Unified Worker Handler
Supports 5 actions:
1. analyze: NSFW detection (CPU)
2. caption: BLIP image captioning (CPU)
3. custom_describe: LLaVA 7B via Ollama (GPU with immediate unload)
4. generate3d: Hunyuan3D-2 geometry + texture generation (GPU with low-VRAM offloading)
5. pipeline: NSFW -> BLIP -> LLaVA -> textured Hunyuan3D in one queued job
"""

import os
import sys

# Hugging Face download configuration (must be set before any HF / transformers / diffusers imports)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

import base64
import gc
import hashlib
import io
import logging
import time
import traceback
from typing import Any, Dict, Generator, Optional, Tuple

from PIL import Image
import httpx
import runpod
import torch
import types

# Compatibility shim: Recent transformers versions call torch.accelerator.current_accelerator()
# which does not exist in PyTorch < 2.6.
if not hasattr(torch, "accelerator"):
    _acc_mod = types.ModuleType("accelerator")
    _acc_mod.current_accelerator = lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _acc_mod.is_available = lambda: torch.cuda.is_available()
    _acc_mod.device_count = lambda: torch.cuda.device_count() if torch.cuda.is_available() else 0
    torch.accelerator = _acc_mod

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

try:
    import transformers
    for _cls_name in ["Dinov2WithRegistersConfig", "Dinov2WithRegistersModel", "Dinov2WithRegistersPreTrainedModel"]:
        if not hasattr(transformers, _cls_name):
            class _DummyTransformerClass:
                pass
            setattr(transformers, _cls_name, _DummyTransformerClass)
except Exception:
    pass

try:
    import numpy as np
except ImportError:
    np = None

# Ensure hy3dgen in subfolder can be imported if not installed in site-packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HUNYUAN_DIR = os.path.join(CURRENT_DIR, "3d_generative", "Hunyuan3D-2-main")
if HUNYUAN_DIR not in sys.path:
    sys.path.insert(0, HUNYUAN_DIR)

# Compatibility shim: Ensure 'modules' resolves to hunyuanpaint.unet.modules for diffusers
try:
    from hy3dgen.texgen.hunyuanpaint.unet import modules as _unet_modules
    sys.modules.setdefault("modules", _unet_modules)
except Exception:
    pass

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
HUNYUAN_SHAPE_MODEL = os.getenv("HUNYUAN_SHAPE_MODEL", "tencent/Hunyuan3D-2.1")
HUNYUAN_SHAPE_SUBFOLDER = os.getenv("HUNYUAN_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-1")
HUNYUAN_TEX_MODEL = os.getenv("HUNYUAN_TEX_MODEL", "tencent/Hunyuan3D-2")
HUNYUAN_SHAPE_USE_SAFETENSORS = not HUNYUAN_SHAPE_MODEL.endswith("Hunyuan3D-2.1")
HUNYUAN_USE_FLASHVDM = "turbo" in HUNYUAN_SHAPE_SUBFOLDER.lower()

DEFAULT_OCTREE_RESOLUTION = 384
DEFAULT_INFERENCE_STEPS = 5 if HUNYUAN_USE_FLASHVDM else 50
DEFAULT_FACE_COUNT = 200000

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
        has_alpha = "A" in pil_img.getbands() or "transparency" in pil_img.info
        pil_img = pil_img.convert("RGBA" if has_alpha else "RGB")
    except Exception as e:
        raise ValueError(f"Invalid image format or corrupted image: {str(e)}")

    return pil_img, image_bytes


def prepare_shape_image(image: Image.Image, rembg) -> Image.Image:
    """Keep a supplied cutout intact; infer alpha only for opaque inputs."""
    if image.mode == "RGBA":
        alpha_min, alpha_max = image.getchannel("A").getextrema()
        if alpha_max == 0:
            raise ValueError("Image alpha channel is fully transparent.")
        if alpha_min < 255:
            return image
    return rembg(image)


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

    def load_shape_pipeline(self):
        """Lazy load rembg and shape pipeline with CPU offloading."""
        if self.hunyuan_worker is None:
            self.hunyuan_worker = {}

        if "rembg" not in self.hunyuan_worker or self.hunyuan_worker["rembg"] is None:
            logger.info("Loading BackgroundRemover (rembg)...")
            from hy3dgen.rembg import BackgroundRemover
            self.hunyuan_worker["rembg"] = BackgroundRemover()

        if "shape_pipeline" not in self.hunyuan_worker or self.hunyuan_worker["shape_pipeline"] is None:
            logger.info(
                f"Loading shape model {HUNYUAN_SHAPE_MODEL} (subfolder: {HUNYUAN_SHAPE_SUBFOLDER})..."
            )
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN_SHAPE_MODEL,
                subfolder=HUNYUAN_SHAPE_SUBFOLDER,
                use_safetensors=HUNYUAN_SHAPE_USE_SAFETENSORS,
                device="cpu",
            )
            if HUNYUAN_USE_FLASHVDM:
                try:
                    topk = "merge" if "mini" in HUNYUAN_SHAPE_MODEL.lower() else "mean"
                    shape_pipeline.enable_flashvdm(topk_mode=topk, mc_algo="mc")
                except Exception as e:
                    logger.warning(f"Could not enable flashvdm: {e}")
            self.hunyuan_worker["shape_pipeline"] = shape_pipeline

        clean_vram()
        return self.hunyuan_worker["rembg"], self.hunyuan_worker["shape_pipeline"]

    @staticmethod
    def offload_tex_pipeline(tex_pipeline):
        """Offload Hunyuan3DPaintPipeline sub-models to CPU and clean VRAM."""
        ModelManager.move_tex_pipeline(tex_pipeline, "cpu")
        clean_vram()

    @staticmethod
    def move_tex_pipeline(tex_pipeline, device: str):
        """Move Hunyuan3DPaintPipeline to target device."""
        if tex_pipeline is not None:
            if hasattr(tex_pipeline, "to"):
                try:
                    tex_pipeline.to(device)
                    return
                except Exception as e:
                    logger.warning(f"Failed to move tex_pipeline to {device}: {e}")
            if hasattr(tex_pipeline, "models"):
                for model_name, model_obj in tex_pipeline.models.items():
                    if hasattr(model_obj, "to"):
                        try:
                            model_obj.to(device)
                        except Exception as e:
                            logger.warning(f"Failed to move {model_name} to {device}: {e}")
                    elif hasattr(model_obj, "pipeline") and model_obj.pipeline is not None:
                        try:
                            model_obj.pipeline.to(device)
                        except Exception as e:
                            logger.warning(f"Failed to move {model_name} to {device}: {e}")

    def load_tex_pipeline(self):
        """Lazy load texture pipeline with CPU offloading."""
        if self.hunyuan_worker is None:
            self.hunyuan_worker = {}

        if "tex_pipeline" not in self.hunyuan_worker or self.hunyuan_worker["tex_pipeline"] is None:
            logger.info(f"Loading texture model {HUNYUAN_TEX_MODEL}...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            try:
                tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_TEX_MODEL)
                self.offload_tex_pipeline(tex_pipeline)
                self.hunyuan_worker["tex_pipeline"] = tex_pipeline
                logger.info("Texture pipeline loaded and offloaded to CPU.")
            except Exception as e:
                logger.error(f"Failed to load texture model: {e}\n{traceback.format_exc()}")
                self.hunyuan_worker["tex_pipeline"] = None

        return self.hunyuan_worker.get("tex_pipeline")

    def load_hunyuan(self, load_texture: bool = True):
        """Load Hunyuan3D pipelines with CPU offloading."""
        rembg, shape_pipeline = self.load_shape_pipeline()
        tex_pipeline = self.load_tex_pipeline() if load_texture else None
        return {
            "rembg": rembg,
            "shape_pipeline": shape_pipeline,
            "tex_pipeline": tex_pipeline,
        }

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
    pil_img = pil_img.convert("RGB")

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
    pil_img = pil_img.convert("RGB")
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
    octree_res = int(job_input.get("octree_resolution", DEFAULT_OCTREE_RESOLUTION))
    steps = int(job_input.get("num_inference_steps", DEFAULT_INFERENCE_STEPS))
    guidance = float(job_input.get("guidance_scale", 5.0))
    face_count = int(job_input.get("face_count", DEFAULT_FACE_COUNT))
    enable_texture = bool(job_input.get("texture", True))

    logger.info(
        "Shape config: model=%s subfolder=%s decoder=%s seed=%d octree=%d steps=%d guidance=%s faces=%d",
        HUNYUAN_SHAPE_MODEL,
        HUNYUAN_SHAPE_SUBFOLDER,
        "flashvdm" if HUNYUAN_USE_FLASHVDM else "standard",
        seed,
        octree_res,
        steps,
        guidance,
        face_count,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    yield {
        "type": "progress",
        "stage": "worker_init",
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "vram": get_vram_info(),
    }

    hunyuan = model_manager.load_hunyuan(load_texture=enable_texture)
    rembg = hunyuan["rembg"]
    shape_pipeline = hunyuan["shape_pipeline"]
    tex_pipeline = hunyuan.get("tex_pipeline")

    # 1. Preprocess & background removal
    yield {
        "type": "progress",
        "stage": "preprocess",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }
    t_prep_start = time.time()
    image_no_bg = prepare_shape_image(pil_img, rembg)
    prep_ms = int((time.time() - t_prep_start) * 1000)

    # 2. Shape generation
    t_shape_start = time.time()
    logger.info(f"Moving shape pipeline to {device}...")
    shape_pipeline.to(device)

    yield {
        "type": "progress",
        "stage": "shape_generation",
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "vram": get_vram_info(),
    }

    mesh = None
    try:
        gen_dev = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
        generator = torch.Generator(device=gen_dev).manual_seed(seed)
        outputs = shape_pipeline(
            image=image_no_bg,
            generator=generator,
            octree_resolution=octree_res,
            num_inference_steps=steps,
            guidance_scale=guidance,
            mc_algo="mc",
        )
        if outputs and len(outputs) > 0:
            mesh = outputs[0]
    except Exception as e:
        logger.warning(f"Shape generation with FlashVDM encountered an error: {e}")

    def _is_mesh_valid(m) -> bool:
        if m is None:
            return False
        if hasattr(m, "vertices"):
            v = getattr(m, "vertices")
            if type(v).__name__ in ("ndarray", "list", "tuple", "Tensor"):
                if len(v) == 0:
                    return False
        if hasattr(m, "faces"):
            f = getattr(m, "faces")
            if type(f).__name__ in ("ndarray", "list", "tuple", "Tensor"):
                if len(f) == 0:
                    return False
        return True

    # Fallback to standard volume decoding if FlashVDM produced empty mesh or failed
    if not _is_mesh_valid(mesh):
        logger.warning("Shape generation returned an empty mesh; retrying once with the standard decoder...")
        try:
            if HUNYUAN_USE_FLASHVDM:
                shape_pipeline.enable_flashvdm(enabled=False)
            gen_dev = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
            generator = torch.Generator(device=gen_dev).manual_seed(seed)
            outputs = shape_pipeline(
                image=image_no_bg,
                generator=generator,
                octree_resolution=octree_res,
                num_inference_steps=steps,
                guidance_scale=guidance,
                mc_algo="mc",
            )
            if outputs and len(outputs) > 0:
                mesh = outputs[0]
        except Exception as e2:
            logger.error(f"Fallback shape generation also failed: {e2}")
        finally:
            if HUNYUAN_USE_FLASHVDM:
                try:
                    topk = "merge" if "mini" in HUNYUAN_SHAPE_MODEL.lower() else "mean"
                    shape_pipeline.enable_flashvdm(topk_mode=topk, mc_algo="mc")
                except Exception:
                    pass

    logger.info("Offloading shape pipeline back to CPU...")
    shape_pipeline.to("cpu")
    clean_vram()

    shape_ms = int((time.time() - t_shape_start) * 1000)

    if not _is_mesh_valid(mesh):
        raise RuntimeError("Hunyuan3D shape generation returned empty mesh.")

    # 3. Mesh cleanup and reduction
    yield {
        "type": "progress",
        "stage": "mesh_cleanup",
        "elapsed_ms": int((time.time() - start_time) * 1000),
    }
    t_clean_start = time.time()
    try:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=face_count)
    except Exception as e:
        logger.warning(
            f"Mesh post-processing encountered an error: {e}. Keeping existing mesh."
        )
    clean_ms = int((time.time() - t_clean_start) * 1000)

    # 4. Texture generation
    tex_ms = 0
    if enable_texture:
        if tex_pipeline is None:
            yield {
                "type": "progress",
                "stage": "loading_texture_pipeline",
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "vram": get_vram_info(),
            }
            tex_pipeline = model_manager.load_tex_pipeline()

        if tex_pipeline is not None:
            t_tex_start = time.time()
            try:
                model_manager.move_tex_pipeline(tex_pipeline, device)
                yield {
                    "type": "progress",
                    "stage": "texture_generation",
                    "elapsed_ms": int((time.time() - start_time) * 1000),
                    "vram": get_vram_info(),
                }
                textured_mesh = tex_pipeline(mesh, image_no_bg)
                if (
                    textured_mesh is not None
                    and getattr(getattr(textured_mesh, "visual", None), "kind", None) == "texture"
                ):
                    mesh = textured_mesh
                    logger.info("Texture generation completed successfully. Textured mesh ready for GLB export.")
                else:
                    raise RuntimeError("Texture pipeline returned a mesh without texture data.")
            except Exception as e:
                logger.error(
                    f"Texture generation encountered an error: {e}\n{traceback.format_exc()}"
                )
                raise RuntimeError(
                    "Texture generation failed; refusing to return an untextured model."
                ) from e
            finally:
                model_manager.offload_tex_pipeline(tex_pipeline)
                clean_vram()
            tex_ms = int((time.time() - t_tex_start) * 1000)
        else:
            raise RuntimeError(
                "Texture pipeline is unavailable; refusing to return an untextured model."
            )

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


def handle_pipeline(
    job_input: Dict[str, Any], start_time: float
) -> Generator[Dict[str, Any], None, None]:
    """Run moderation, captioning, visual analysis, and textured 3D in one queued job."""
    moderation = None
    for event in handle_analyze(job_input, start_time):
        if event.get("type") == "result":
            moderation = event["data"]
        else:
            if event.get("type") == "progress":
                event = {**event, "stage": f"moderation.{event['stage']}"}
            yield event

    if not moderation:
        raise RuntimeError("NSFW stage returned no result.")
    if moderation.get("is_nsfw"):
        yield {
            "type": "error",
            "stage": "moderation",
            "error": "Image violates NSFW policy.",
            "data": moderation,
        }
        return

    caption_input = {**job_input, "max_new_tokens": 30, "min_new_tokens": 5}
    caption = None
    for event in handle_caption(caption_input, start_time):
        if event.get("type") == "result":
            caption = event["data"]
        else:
            if event.get("type") == "progress":
                event = {**event, "stage": f"caption.{event['stage']}"}
            yield event

    if not caption:
        raise RuntimeError("Caption stage returned no result.")

    short_text = caption.get("caption") or "the main object"
    analysis_request = job_input.get("prompt") or (
        "Phân tích vật thể để dựng mô hình 3D: hình khối, vật liệu, các mặt khuất, "
        "chi tiết bề mặt và ánh sáng."
    )
    describe_input = {
        **job_input,
        "prompt": f"Mô tả ngắn: {short_text}. {analysis_request}",
    }
    description = None
    for event in handle_custom_describe(describe_input, start_time):
        if event.get("type") == "result":
            description = event["data"]
        else:
            if event.get("type") == "progress":
                event = {**event, "stage": f"analysis.{event['stage']}"}
            yield event

    if not description:
        raise RuntimeError("Visual analysis stage returned no result.")

    yield {
        "type": "result",
        "action": "pipeline",
        "data": {
            "moderation": moderation,
            "description": caption,
            "rich_prompt": description.get("description"),
            "rich_prompt_en": description.get("description_en"),
        },
    }

    generate_input = {**job_input, "texture": True, "texture_resolution": 1024}
    for event in handle_generate3d(generate_input, start_time):
        if event.get("type") == "progress":
            event = {**event, "stage": f"generate3d.{event['stage']}"}
        yield event


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
            "error": "Missing required field 'action' in input. Expected: analyze, caption, custom_describe, generate3d, pipeline.",
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
        elif action == "pipeline":
            yield from handle_pipeline(job_input, start_time)
        else:
            yield {
                "type": "error",
                "stage": "validate_input",
                "error": f"Invalid action '{action}'. Supported actions: analyze, caption, custom_describe, generate3d, pipeline.",
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
    runpod.serverless.start({"handler": handler})
