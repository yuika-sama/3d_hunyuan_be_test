# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import types
import torch

# Compatibility shim: Recent transformers versions call torch.accelerator.current_accelerator()
# which does not exist in PyTorch < 2.6.
if not hasattr(torch, "accelerator"):
    _acc_mod = types.ModuleType("accelerator")
    _acc_mod.current_accelerator = lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _acc_mod.is_available = lambda: torch.cuda.is_available()
    _acc_mod.device_count = lambda: torch.cuda.device_count() if torch.cuda.is_available() else 0
    torch.accelerator = _acc_mod

# Compatibility shim: Diffusers >= 0.32 autoencoder_rae imports Dinov2WithRegistersConfig
# which is absent in older transformers.
try:
    import transformers
    for _cls_name in ["Dinov2WithRegistersConfig", "Dinov2WithRegistersModel", "Dinov2WithRegistersPreTrainedModel"]:
        if not hasattr(transformers, _cls_name):
            class _DummyTransformerClass:
                pass
            setattr(transformers, _cls_name, _DummyTransformerClass)
except Exception:
    pass