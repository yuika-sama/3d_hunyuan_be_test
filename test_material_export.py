"""CPU regression tests for the color-only Hunyuan Paint GLB exporter."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import unittest

import numpy as np
from PIL import Image
import trimesh

MODULE = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen/differentiable_renderer/mesh_utils.py"
spec = importlib.util.spec_from_file_location("paint_mesh_utils", MODULE)
mesh_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_utils)


def fixture():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]], process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [0, 1]])
    pixels = np.array([[[190, 80, 30], [220, 140, 50]], [[90, 40, 10], [255, 200, 100]]], dtype=np.uint8)
    return mesh, Image.fromarray(pixels)


class TestPaintMaterial(unittest.TestCase):
    def test_color_only_material_is_explicitly_nonmetallic(self):
        mesh, image = fixture()
        output = mesh_utils.save_mesh(mesh, image)
        self.assertIs(output, mesh)
        self.assertIsInstance(output.visual.material, trimesh.visual.material.PBRMaterial)
        self.assertEqual(output.visual.material.metallicFactor, 0.0)
        self.assertEqual(output.visual.material.roughnessFactor, 0.9)
        self.assertEqual(output.visual.material.alphaMode, "OPAQUE")

    def test_glb_embeds_the_texture_and_explicit_material_factors(self):
        mesh, image = fixture()
        binary = mesh_utils.save_mesh(mesh, image).export(file_type="glb")
        self.assertEqual(struct.unpack_from("<4sII", binary), (b"glTF", 2, len(binary)))
        json_length, kind = struct.unpack_from("<II", binary, 12)
        self.assertEqual(kind, 0x4E4F534A)
        model = json.loads(binary[20:20 + json_length])
        material = model["materials"][0]
        pbr = material["pbrMetallicRoughness"]
        self.assertEqual(pbr["metallicFactor"], 0.0)
        self.assertEqual(pbr["roughnessFactor"], 0.9)
        self.assertEqual(pbr.get("baseColorFactor", [1, 1, 1, 1]), [1, 1, 1, 1])
        self.assertIn("baseColorTexture", pbr)
        self.assertEqual(material["alphaMode"], "OPAQUE")
        self.assertIn("bufferView", model["images"][0])
        loaded = trimesh.load(io.BytesIO(binary), file_type="glb", force="mesh", process=False)
        np.testing.assert_array_equal(np.array(loaded.visual.material.baseColorTexture), np.array(image))

    def test_export_keeps_geometry_uvs_and_texture_pixels(self):
        mesh, image = fixture()
        vertices, faces, uv = mesh.vertices.copy(), mesh.faces.copy(), mesh.visual.uv.copy()
        output = mesh_utils.save_mesh(mesh, image)
        np.testing.assert_array_equal(output.vertices, vertices)
        np.testing.assert_array_equal(output.faces, faces)
        np.testing.assert_array_equal(output.visual.uv, uv)
        material = output.visual.material
        texture = getattr(material, "baseColorTexture", getattr(material, "image", None))
        np.testing.assert_array_equal(np.array(texture), np.array(image))


if __name__ == "__main__":
    unittest.main()
