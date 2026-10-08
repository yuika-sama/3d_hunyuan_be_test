"""
Compatibility shim module for Hunyuan3D texture generation.
Diffusers reads model_index.json which specifies:
    "unet": ["modules", "UNet2p5DConditionModel"]
When diffusers calls `importlib.import_module("modules")`, this module resolves it
and exposes all classes/functions from hy3dgen.texgen.hunyuanpaint.unet.modules.
"""
import sys

try:
    from hy3dgen.texgen.hunyuanpaint.unet.modules import *
    from hy3dgen.texgen.hunyuanpaint.unet import modules as _mod
    sys.modules.setdefault("modules", sys.modules[__name__])
except Exception:
    # Graceful fallback in environments where heavy submodules are mocked or missing optional deps
    pass
