"""
Preload helper for Runpod Serverless Worker.
Pre-downloads rembg, Hunyuan3D model weights, and NSFW models to local cache or Network Volume.
Ensures only the selected shape checkpoint is fetched.
"""
import os
import sys

# Configure volume-based cache paths if Global Volume or Network Volume is mounted
def resolve_volume_path() -> str:
    custom = os.getenv("RUNPOD_VOLUME_PATH")
    if custom and os.path.isdir(custom):
        return custom
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume"
    if os.path.isdir("/workspace") and os.access("/workspace", os.W_OK):
        return "/workspace"
    return ""

VOLUME_PATH = resolve_volume_path()
if VOLUME_PATH:
    os.environ.setdefault("HF_HOME", os.path.join(VOLUME_PATH, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(VOLUME_PATH, "cache", "torch"))
    os.environ.setdefault("U2NET_HOME", os.path.join(VOLUME_PATH, "cache", "u2net"))
    os.environ.setdefault("HY3DGEN_MODELS", os.path.join(VOLUME_PATH, "models"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(VOLUME_PATH, "cache", "triton"))
    for _p in (
        os.environ["HF_HOME"],
        os.environ["TORCH_HOME"],
        os.environ["U2NET_HOME"],
        os.environ["HY3DGEN_MODELS"],
        os.environ["TRITON_CACHE_DIR"],
    ):
        try:
            os.makedirs(_p, exist_ok=True)
        except OSError:
            pass

# Ensure Hugging Face environment variables are set before any HF imports
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
if os.getenv("HF_TOKEN") and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HUNYUAN_DIR = os.path.join(CURRENT_DIR, "3d_generative", "Hunyuan3D-2-main")
HUNYUAN_SHAPE_MODEL = os.getenv("HUNYUAN_SHAPE_MODEL", "tencent/Hunyuan3D-2.1")
HUNYUAN_SHAPE_SUBFOLDER = os.getenv("HUNYUAN_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-1")
HUNYUAN_SHAPE_USE_SAFETENSORS = not HUNYUAN_SHAPE_MODEL.endswith("Hunyuan3D-2.1")
if HUNYUAN_DIR not in sys.path:
    sys.path.insert(0, HUNYUAN_DIR)

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
    from hy3dgen.texgen.hunyuanpaint.unet import modules as _unet_modules
    sys.modules.setdefault("modules", _unet_modules)
except Exception:
    pass


def preload_shape_checkpoint():
    """Download shape weights without instantiating the multi-GB model."""
    from huggingface_hub import snapshot_download

    extension = "safetensors" if HUNYUAN_SHAPE_USE_SAFETENSORS else "ckpt"
    snapshot_download(
        repo_id=HUNYUAN_SHAPE_MODEL,
        allow_patterns=[
            f"{HUNYUAN_SHAPE_SUBFOLDER}/*.yaml",
            f"{HUNYUAN_SHAPE_SUBFOLDER}/*.json",
            f"{HUNYUAN_SHAPE_SUBFOLDER}/*.{extension}",
        ],
        max_workers=2,
    )


def preload(full_texture: bool = False, include_nsfw: bool = False):
    failures = []
    active_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print("=" * 60)
    print(f" [PRELOAD] Starting model preload (full_texture={full_texture}, include_nsfw={include_nsfw})...")
    print(f" [PRELOAD] Active HF Cache directory: {active_cache}")
    if VOLUME_PATH and os.path.isdir(VOLUME_PATH):
        print(f" [PRELOAD] Persistent RunPod Volume: {VOLUME_PATH}")
    else:
        print(" [PRELOAD] Running on container disk (RunPod Volume not detected).")
    print("=" * 60)

    print("[PRELOAD] 1/4 Checking / downloading BackgroundRemover model (u2net.onnx)...")
    try:
        from hy3dgen.rembg import BackgroundRemover
        BackgroundRemover()
        print("[PRELOAD] BackgroundRemover ready.")
    except Exception as e:
        print(f"[PRELOAD WARNING] BackgroundRemover failed: {e}")
        failures.append("BackgroundRemover")

    print("[PRELOAD] 2/4 Checking / downloading Hunyuan3D shape model...")
    try:
        preload_shape_checkpoint()
        print("[PRELOAD] Hunyuan3D shape model ready.")
    except Exception as e:
        print(f"[PRELOAD WARNING] Hunyuan3D shape model failed: {e}")
        failures.append("shape")

    if full_texture:
        print("[PRELOAD] 3/4 Checking / downloading Hunyuan3D texture model...")
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            Hunyuan3DPaintPipeline.resolve_config(
                os.getenv("HUNYUAN_TEX_MODEL", "tencent/Hunyuan3D-2"),
                subfolder=os.getenv("HUNYUAN_TEX_SUBFOLDER", "hunyuan3d-paint-v2-0"),
            )
            print("[PRELOAD] Hunyuan3D texture model ready.")
        except Exception as e:
            print(f"[PRELOAD WARNING] Hunyuan3D texture model failed during preload: {e}")
            import traceback
            traceback.print_exc()
            failures.append("texture")

    if include_nsfw:
        print("[PRELOAD] 4/4 Checking / downloading NSFW classification model...")
        try:
            from transformers import AutoImageProcessor, SiglipForImageClassification
            nsfw_name = os.getenv("NSFW_MODEL_NAME", "strangerguardhf/nsfw_image_detection")
            AutoImageProcessor.from_pretrained(nsfw_name)
            SiglipForImageClassification.from_pretrained(nsfw_name)
            print("[PRELOAD] NSFW classification model ready.")
        except Exception as e:
            print(f"[PRELOAD WARNING] NSFW model preload failed: {e}")
            failures.append("nsfw")

    if failures:
        raise RuntimeError(f"Model preload failed: {', '.join(failures)}")

    print("=" * 60)
    print(" [PRELOAD] Preload process completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    is_full = "--full" in sys.argv
    is_nsfw = "--nsfw" in sys.argv
    preload(full_texture=is_full, include_nsfw=is_nsfw)
