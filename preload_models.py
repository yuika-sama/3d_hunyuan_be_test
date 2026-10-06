"""
Preload helper for Runpod Serverless Worker.
Pre-downloads rembg and Hunyuan3D model weights to local cache.
Ensures only needed safetensors files are fetched.
"""
import os
import sys

# Ensure Hugging Face environment variables are set before any HF imports
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HUNYUAN_DIR = os.path.join(CURRENT_DIR, "3dgen", "Hunyuan3D-2-main")
if HUNYUAN_DIR not in sys.path:
    sys.path.insert(0, HUNYUAN_DIR)


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

    print("[PRELOAD] 2/3 Checking / downloading Hunyuan3D shape model (safetensors only)...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini-turbo",
            use_safetensors=True,
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
            if hasattr(tex_pipeline, "models"):
                for m in tex_pipeline.models.values():
                    if hasattr(m, "pipeline") and m.pipeline is not None:
                        m.pipeline.to("cpu")
            print("[PRELOAD] Hunyuan3D texture model ready.")
        except Exception as e:
            print(f"[PRELOAD WARNING] Hunyuan3D texture model failed: {e}")

    print("=" * 60)
    print(" [PRELOAD] Preload process completed.")
    print("=" * 60)


if __name__ == "__main__":
    is_full = "--full" in sys.argv
    preload(full_texture=is_full)
