"""Compare the Linux worker's compiled inpaint with the portable implementation."""
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def test_native_inpaint(monkeypatch):
    root = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer"
    monkeypatch.syspath_prepend(str(root))
    import mesh_processor as native

    assert any(native.__file__.endswith(ext) for ext in EXTENSION_SUFFIXES)
    spec = spec_from_file_location("python_inpaint", root / "mesh_processor.py")
    reference = module_from_spec(spec)
    spec.loader.exec_module(reference)

    tex = np.ones((5, 8, 3), np.float32)[:, ::2]
    mask = np.zeros(tex.shape[:2], np.uint8)
    mask[-1, 0] = 255
    vertices = np.eye(3, dtype=np.float32)
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    inputs = (tex, mask, vertices, uv, faces, faces)
    actual = native.meshVerticeInpaint(*inputs)
    expected = reference.meshVerticeInpaint(*inputs)
    for result, baseline in zip(actual, expected):
        np.testing.assert_allclose(result, baseline)
    assert actual[1][0, 0] == 255
    with pytest.raises(ValueError, match="index"):
        native.meshVerticeInpaint(tex, mask, vertices, uv, faces + 9, faces)
