"""
Runpod Serverless Unified Worker Handler
Supports 3 actions:
1. analyze: NSFW detection (CPU)
2. generate3d: Hunyuan3D-2 geometry + texture generation (GPU with low-VRAM offloading)
3. pipeline: NSFW -> textured Hunyuan3D in one queued job
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
NSFW_MODEL_NAME = os.getenv("NSFW_MODEL_NAME", "strangerguardhf/nsfw_image_detection")
HUNYUAN_SHAPE_MODEL = os.getenv("HUNYUAN_SHAPE_MODEL", "tencent/Hunyuan3D-2.1")
HUNYUAN_SHAPE_SUBFOLDER = os.getenv("HUNYUAN_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-1")
HUNYUAN_TEX_MODEL = os.getenv("HUNYUAN_TEX_MODEL", "tencent/Hunyuan3D-2")
HUNYUAN_TEX_SUBFOLDER = os.getenv("HUNYUAN_TEX_SUBFOLDER", "hunyuan3d-paint-v2-0")
HUNYUAN_SHAPE_USE_SAFETENSORS = not HUNYUAN_SHAPE_MODEL.endswith("Hunyuan3D-2.1")
HUNYUAN_USE_FLASHVDM = "turbo" in HUNYUAN_SHAPE_SUBFOLDER.lower()

DEFAULT_OCTREE_RESOLUTION = int(os.getenv("DEFAULT_OCTREE_RESOLUTION", "384"))
DEFAULT_INFERENCE_STEPS = 5 if HUNYUAN_USE_FLASHVDM else 50
DEFAULT_FACE_COUNT = int(os.getenv("DEFAULT_FACE_COUNT", "40000"))
DEFAULT_TEXTURE_RESOLUTION = int(os.getenv("DEFAULT_TEXTURE_RESOLUTION", "2048"))
DEFAULT_BAKE_EXP = int(os.getenv("DEFAULT_BAKE_EXP", "4"))

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
    """Keep a supplied cutout intact; infer alpha only for opaque inputs; ensure RGBA."""
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        if alpha_max == 0:
            raise ValueError("Image alpha channel is fully transparent.")
        if alpha_min < 255:
            return image
    res = rembg(image)
    if res.mode != "RGBA":
        res = res.convert("RGBA")
    return res


def chunk_bytes(data: bytes, chunk_size: int = CHUNK_SIZE_BYTES) -> list:
    """Split bytes into chunks of specified size."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


class ModelManager:
    """Singleton-style manager for lazy loading and lifecycle management of ML models."""

    _instance = None

    def __init__(self):
        self.nsfw_processor = None
        self.nsfw_model = None
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

    def load_tex_pipeline(self, subfolder: Optional[str] = None):
        """Lazy load texture pipeline with CPU offloading."""
        if self.hunyuan_worker is None:
            self.hunyuan_worker = {}

        target_subfolder = subfolder or HUNYUAN_TEX_SUBFOLDER
        current_tex_pipe = self.hunyuan_worker.get("tex_pipeline")
        current_subfolder = getattr(current_tex_pipe, "_loaded_subfolder", None)

        if current_tex_pipe is not None and current_subfolder == target_subfolder:
            return current_tex_pipe

        logger.info(f"Loading texture model {HUNYUAN_TEX_MODEL} (subfolder: {target_subfolder})...")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        try:
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                HUNYUAN_TEX_MODEL, subfolder=target_subfolder
            )
            tex_pipeline._loaded_subfolder = target_subfolder
            self.offload_tex_pipeline(tex_pipeline)
            self.hunyuan_worker["tex_pipeline"] = tex_pipeline
            logger.info("Texture pipeline loaded and offloaded to CPU.")
        except Exception as e:
            logger.error(f"Failed to load texture model: {e}\n{traceback.format_exc()}")
            if target_subfolder != "hunyuan3d-paint-v2-0-turbo":
                logger.warning("Attempting fallback to 'hunyuan3d-paint-v2-0-turbo'...")
                try:
                    tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                        HUNYUAN_TEX_MODEL, subfolder="hunyuan3d-paint-v2-0-turbo"
                    )
                    tex_pipeline._loaded_subfolder = "hunyuan3d-paint-v2-0-turbo"
                    self.offload_tex_pipeline(tex_pipeline)
                    self.hunyuan_worker["tex_pipeline"] = tex_pipeline
                    logger.info("Fallback turbo texture pipeline loaded and offloaded to CPU.")
                except Exception as e2:
                    logger.error(f"Fallback turbo texture model also failed: {e2}")
                    self.hunyuan_worker["tex_pipeline"] = None
            else:
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

    # Texture configuration parameters
    tex_res = int(
        job_input.get("texture_resolution")
        or job_input.get("texture_size")
        or DEFAULT_TEXTURE_RESOLUTION
    )
    tex_subfolder = (
        job_input.get("texture_subfolder")
        or job_input.get("subfolder")
        or HUNYUAN_TEX_SUBFOLDER
    )
    bake_exp = int(job_input.get("bake_exp", DEFAULT_BAKE_EXP))

    texture_imgs = []
    if enable_texture:
        mv_inputs = job_input.get("texture_images_base64") or job_input.get("texture_images")
        if mv_inputs and isinstance(mv_inputs, list):
            for t_item in mv_inputs:
                t_dec, _ = decode_base64_image(t_item)
                texture_imgs.append(t_dec)
        else:
            single_tex = job_input.get("texture_image_base64") or job_input.get("texture_image")
            if single_tex:
                t_dec, _ = decode_base64_image(single_tex)
                texture_imgs.append(t_dec)
            else:
                texture_imgs.append(pil_img)

    logger.info(
        "Shape config: model=%s subfolder=%s decoder=%s seed=%d octree=%d steps=%d guidance=%s faces=%d tex_res=%d bake_exp=%d",
        HUNYUAN_SHAPE_MODEL,
        HUNYUAN_SHAPE_SUBFOLDER,
        "flashvdm" if HUNYUAN_USE_FLASHVDM else "standard",
        seed,
        octree_res,
        steps,
        guidance,
        face_count,
        tex_res,
        bake_exp,
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
    texture_images_no_bg = []
    for t_img in texture_imgs:
        if t_img is pil_img:
            texture_images_no_bg.append(image_no_bg)
        else:
            texture_images_no_bg.append(prepare_shape_image(t_img, rembg))
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
            tex_pipeline = model_manager.load_tex_pipeline(subfolder=tex_subfolder)

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
                tex_input = texture_images_no_bg[0] if len(texture_images_no_bg) == 1 else texture_images_no_bg
                textured_mesh = tex_pipeline(
                    mesh,
                    tex_input,
                    render_size=tex_res,
                    texture_size=tex_res,
                    bake_exp=bake_exp,
                )
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
    """Run moderation and textured 3D generation in one queued job."""
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

    texture_moderation = None
    if job_input.get("texture_image_base64") or job_input.get("texture_image"):
        texture_input = {
            **job_input,
            "image_base64": job_input.get("texture_image_base64"),
            "image": job_input.get("texture_image"),
        }
        for event in handle_analyze(texture_input, start_time):
            if event.get("type") == "result":
                texture_moderation = event["data"]
            elif event.get("type") == "progress":
                yield {**event, "stage": f"texture_moderation.{event['stage']}"}

        if not texture_moderation:
            raise RuntimeError("Texture NSFW stage returned no result.")
        if texture_moderation.get("is_nsfw"):
            yield {
                "type": "error",
                "stage": "texture_moderation",
                "error": "Texture image violates NSFW policy.",
                "data": texture_moderation,
            }
            return
    else:
        texture_moderation = moderation

    yield {
        "type": "result",
        "action": "pipeline",
        "data": {"moderation": moderation, "texture_moderation": texture_moderation},
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
            "error": "Missing required field 'action' in input. Expected: analyze, generate3d, pipeline.",
            "timings": {"total_ms": 0},
        }
        return

    logger.info(f"Starting job with action='{action}'")

    try:
        if action == "analyze":
            yield from handle_analyze(job_input, start_time)
        elif action == "generate3d":
            yield from handle_generate3d(job_input, start_time)
        elif action == "pipeline":
            yield from handle_pipeline(job_input, start_time)
        else:
            yield {
                "type": "error",
                "stage": "validate_input",
                "error": f"Invalid action '{action}'. Supported actions: analyze, generate3d, pipeline.",
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
