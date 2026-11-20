"""Resolve the same shape weights for preload and inference."""
import logging
import os
from pathlib import Path


def validate_checkpoint(path):
    from safetensors import SafetensorError, safe_open

    try:
        if path.suffix == ".safetensors":
            with safe_open(str(path), framework="pt", device="cpu") as tensors:
                if not tensors.keys():
                    raise ValueError("checkpoint contains no tensors")
        elif path.stat().st_size == 0:
            raise ValueError("empty checkpoint")
    except (OSError, ValueError, RuntimeError, SafetensorError) as exc:
        raise ValueError(f"Invalid shape checkpoint {path}: {exc}") from exc


def smart_load_model(model_path, subfolder, use_safetensors, variant):
    local_model = Path(model_path).expanduser()
    explicit = local_model.is_absolute() or local_model.is_dir()
    root = local_model if explicit else Path(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")).expanduser() / model_path
    folder = root / subfolder
    config = folder / "config.yaml"
    extension = "safetensors" if use_safetensors else "ckpt"
    filename = f"model.{variant}.{extension}" if variant else f"model.{extension}"
    weights = folder / filename
    if explicit:
        if not config.is_file():
            raise FileNotFoundError(config)
        validate_checkpoint(weights)
        return str(config), str(weights)

    def download(name, force=False):
        from huggingface_hub import hf_hub_download

        # local_dir materializes regular files on Global Volumes.
        hf_hub_download(repo_id=model_path, filename=f"{subfolder}/{name}", local_dir=str(root),
                        local_dir_use_symlinks=False, force_download=force)

    if not config.is_file() or config.stat().st_size == 0:
        download(config.name, force=config.exists())
    if not weights.is_file():
        download(filename)
    try:
        validate_checkpoint(weights)
    except ValueError as exc:
        logging.getLogger(__name__).warning("%s. Downloading a fresh copy.", exc)
        download(filename, force=True)
        validate_checkpoint(weights)
    return str(config), str(weights)
