"""
Preload helper for Runpod Serverless Worker.
Pre-downloads rembg and Hunyuan3D model weights to local cache.
Ensures only the selected shape checkpoint is fetched.
"""
import os
import sys

# Ensure Hugging Face environment variables are set before any HF imports
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

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


def preload(full_texture: bool = False):
    print("=" * 60)
    print(f" [PRELOAD] Starting model preload (full_texture={full_texture})...")
    print("=" * 60)

    print("[PRELOAD] 1/3 Checking / downloading BackgroundRemover model (u2net.onnx)...")
    try:
        from hy3dgen.rembg import BackgroundRemover
        BackgroundRemover()
        print("[PRELOAD] BackgroundRemover ready.")
    except Exception as e:
        print(f"[PRELOAD WARNING] BackgroundRemover failed: {e}")

    print("[PRELOAD] 2/3 Checking / downloading Hunyuan3D shape model...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            HUNYUAN_SHAPE_MODEL,
            subfolder=HUNYUAN_SHAPE_SUBFOLDER,
            use_safetensors=HUNYUAN_SHAPE_USE_SAFETENSORS,
            device="cpu",
        )
        print("[PRELOAD] Hunyuan3D shape model ready.")
    except Exception as e:
        print(f"[PRELOAD WARNING] Hunyuan3D shape model failed: {e}")

    if full_texture:
        print("[PRELOAD] 3/3 Checking / downloading Hunyuan3D texture model...")
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
            tex_pipeline.to("cpu")
            print("[PRELOAD] Hunyuan3D texture model ready.")
        except Exception as e:
            print(f"[PRELOAD WARNING] Hunyuan3D texture model failed during preload: {e}")
            import traceback
            traceback.print_exc()
            print("[PRELOAD WARNING] Continuing container startup so worker can handle requests...")

    print("=" * 60)
    print(" [PRELOAD] Preload process completed.")
    print("=" * 60)


if __name__ == "__main__":
    is_full = "--full" in sys.argv
    preload(full_texture=is_full)
