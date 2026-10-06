"""
Unit and Integration Tests for Runpod Serverless Worker
Covers:
- Image decoding and validation (base64, data URIs, error cases)
- Chunking and SHA256 GLB reassembly
- Handler action routing, error reporting, and stream generator events
- Memory management and Ollama model unload calls
"""

import base64
import hashlib
import io
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import runpod_handler
from runpod_handler import (
    CHUNK_SIZE_BYTES,
    chunk_bytes,
    decode_base64_image,
    handler,
)


def create_dummy_image_b64(width: int = 64, height: int = 64, color=(255, 0, 0)) -> str:
    """Helper to create a solid color PNG image encoded in base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestRunpodHandler(unittest.TestCase):
    def setUp(self):
        self.dummy_b64 = create_dummy_image_b64()

    def test_decode_base64_image_valid(self):
        """Test decoding raw base64 string to PIL Image and bytes."""
        pil_img, raw_bytes = decode_base64_image(self.dummy_b64)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.size, (64, 64))
        self.assertIsInstance(raw_bytes, bytes)
        self.assertGreater(len(raw_bytes), 0)

    def test_decode_base64_image_with_data_uri(self):
        """Test decoding image with 'data:image/png;base64,' prefix."""
        uri_str = f"data:image/png;base64,{self.dummy_b64}"
        pil_img, raw_bytes = decode_base64_image(uri_str)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.size, (64, 64))

    def test_decode_base64_image_invalid_inputs(self):
        """Test error handling for empty or invalid image strings."""
        with self.assertRaises(ValueError):
            decode_base64_image("")

        with self.assertRaises(ValueError):
            decode_base64_image(None)

        with self.assertRaises(ValueError):
            decode_base64_image("not_a_valid_base64_image_string_!!@@")

    def test_chunking_and_sha256_reconstruction(self):
        """Test splitting byte stream into 512KB chunks and reconstructing with SHA256 verification."""
        # Create 1.5 MB of random test bytes
        test_data = b"RunpodServerlessGLBChunkTest" * 60000
        original_hash = hashlib.sha256(test_data).hexdigest()

        chunks = chunk_bytes(test_data, chunk_size=CHUNK_SIZE_BYTES)
        expected_chunks = (len(test_data) + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
        self.assertEqual(len(chunks), expected_chunks)

        # Reconstruct
        reconstructed = b"".join(chunks)
        reconstructed_hash = hashlib.sha256(reconstructed).hexdigest()

        self.assertEqual(reconstructed, test_data)
        self.assertEqual(reconstructed_hash, original_hash)

    def test_handler_missing_action(self):
        """Test error event when 'action' is missing from request input."""
        job = {"input": {"image_base64": self.dummy_b64}}
        events = list(handler(job))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["stage"], "validate_input")

    def test_handler_invalid_action(self):
        """Test error event when 'action' is unknown."""
        job = {"input": {"action": "unknown_action_xyz", "image_base64": self.dummy_b64}}
        events = list(handler(job))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("Invalid action", events[0]["error"])

    @patch.object(runpod_handler.model_manager, "load_nsfw")
    def test_handler_analyze_success(self, mock_load_nsfw):
        """Test analyze action stream events with mocked NSFW model."""
        import torch

        mock_processor = MagicMock()
        mock_processor.return_value = MagicMock(to=lambda d: {"pixel_values": torch.zeros((1, 3, 224, 224))})

        mock_model = MagicMock()
        # Mock logits: highest score for index 2 ("Normal")
        mock_model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.05, 0.95, 0.01, 0.02]]))
        mock_load_nsfw.return_value = (mock_processor, mock_model)

        job = {"input": {"action": "analyze", "image_base64": self.dummy_b64}}
        events = list(handler(job))

        stages = [e.get("stage") for e in events if e.get("type") == "progress"]
        self.assertIn("decode_input", stages)
        self.assertIn("nsfw_inference", stages)

        result_event = next(e for e in events if e.get("type") == "result")
        self.assertEqual(result_event["action"], "analyze")
        self.assertTrue(result_event["data"]["ok"])
        self.assertEqual(result_event["data"]["top_label"], "Normal")
        self.assertFalse(result_event["data"]["is_nsfw"])

    @patch.object(runpod_handler.model_manager, "load_blip")
    def test_handler_caption_success(self, mock_load_blip):
        """Test caption action stream events with mocked BLIP model."""
        mock_processor = MagicMock()
        mock_processor.return_value = MagicMock(to=lambda d: {})
        mock_processor.decode.return_value = "a 3D model of a red cube"

        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_load_blip.return_value = (mock_processor, mock_model)

        job = {"input": {"action": "caption", "image_base64": self.dummy_b64, "prompt": "a photo of"}}
        events = list(handler(job))

        result_event = next(e for e in events if e.get("type") == "result")
        self.assertEqual(result_event["action"], "caption")
        self.assertEqual(result_event["data"]["caption"], "a 3D model of a red cube")

    @patch("runpod_handler.httpx.Client")
    @patch.object(runpod_handler.model_manager, "unload_ollama")
    def test_handler_custom_describe_success(self, mock_unload_ollama, mock_client_cls):
        """Test custom_describe action stream events with mocked Ollama HTTP response."""
        mock_client = MagicMock()
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"response": "A detailed red textured cube."}
        mock_client.__enter__.return_value.post.return_value = mock_res
        mock_client_cls.return_value = mock_client

        job = {
            "input": {
                "action": "custom_describe",
                "image_base64": self.dummy_b64,
                "prompt": "Mô tả chi tiết vật thể",
            }
        }
        events = list(handler(job))

        result_event = next(e for e in events if e.get("type") == "result")
        self.assertEqual(result_event["action"], "custom_describe")
        self.assertTrue(result_event["data"]["ok"])
        self.assertIn("description", result_event["data"])
        mock_unload_ollama.assert_called_once()

    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    def test_handler_generate3d_streaming_chunks(self, mock_load_hunyuan):
        """Test generate3d action with mocked pipelines, verifying GLB chunking and SHA256 integrity."""
        # Create a mock mesh that exports 1.2 MB of dummy GLB bytes
        fake_glb_bytes = b"GLBHEADER_MOCK_DATA" * 60000
        original_sha256 = hashlib.sha256(fake_glb_bytes).hexdigest()

        mock_mesh = MagicMock()

        def mock_export(file_obj, file_type="glb"):
            file_obj.write(fake_glb_bytes)

        mock_mesh.export.side_effect = mock_export

        mock_rembg = MagicMock(return_value=Image.new("RGB", (64, 64)))
        mock_shape_pipe = MagicMock()
        mock_shape_pipe.return_value = [mock_mesh]
        mock_tex_pipe = MagicMock()
        mock_tex_pipe.return_value = mock_mesh

        mock_load_hunyuan.return_value = {
            "rembg": mock_rembg,
            "shape_pipeline": mock_shape_pipe,
            "tex_pipeline": mock_tex_pipe,
        }

        mock_shapegen = MagicMock()
        mock_shapegen.FloaterRemover = MagicMock(return_value=lambda m: m)
        mock_shapegen.DegenerateFaceRemover = MagicMock(return_value=lambda m: m)
        mock_shapegen.FaceReducer = MagicMock(return_value=lambda m, max_facenum: m)

        with patch.dict(
            "sys.modules",
            {
                "hy3dgen": MagicMock(),
                "hy3dgen.shapegen": mock_shapegen,
                "hy3dgen.rembg": MagicMock(),
                "hy3dgen.texgen": MagicMock(),
            },
        ):
            job = {
                "input": {
                    "action": "generate3d",
                    "image_base64": self.dummy_b64,
                    "texture": False,
                }
            }
            events = list(handler(job))

        # Check progress events
        progress_stages = [e["stage"] for e in events if e.get("type") == "progress"]
        self.assertIn("shape_generation", progress_stages)
        self.assertIn("glb_export", progress_stages)

        # Check file metadata event
        meta_event = next(e for e in events if e.get("type") == "file_meta")
        self.assertEqual(meta_event["name"], "model.glb")
        self.assertEqual(meta_event["size"], len(fake_glb_bytes))
        self.assertEqual(meta_event["sha256"], original_sha256)

        # Reconstruct chunks
        chunk_events = [e for e in events if e.get("type") == "file_chunk"]
        self.assertEqual(len(chunk_events), meta_event["chunks"])

        reconstructed_bytes = bytearray()
        for c in sorted(chunk_events, key=lambda x: x["index"]):
            reconstructed_bytes.extend(base64.b64decode(c["data"]))

        self.assertEqual(bytes(reconstructed_bytes), fake_glb_bytes)
        self.assertEqual(
            hashlib.sha256(reconstructed_bytes).hexdigest(), original_sha256
        )

        # Check done event
        done_event = next(e for e in events if e.get("type") == "done")
        self.assertIn("timings", done_event)
        self.assertIn("total_ms", done_event["timings"])

    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    @patch.object(runpod_handler.model_manager, "move_tex_pipeline")
    @patch.object(runpod_handler.model_manager, "offload_tex_pipeline")
    def test_handler_generate3d_with_texture_success(
        self, mock_offload, mock_move, mock_load_hunyuan
    ):
        """Test generate3d action with texture=True, verifying texture stage and offloading."""
        fake_glb_bytes = b"GLB_TEXTURED_DATA" * 1000
        mock_mesh = MagicMock()
        mock_mesh.export.side_effect = lambda f, file_type: f.write(fake_glb_bytes)

        mock_rembg = MagicMock(return_value=Image.new("RGB", (64, 64)))
        mock_shape_pipe = MagicMock()
        mock_shape_pipe.return_value = [mock_mesh]
        mock_tex_pipe = MagicMock()
        mock_tex_pipe.return_value = mock_mesh

        mock_load_hunyuan.return_value = {
            "rembg": mock_rembg,
            "shape_pipeline": mock_shape_pipe,
            "tex_pipeline": mock_tex_pipe,
        }

        mock_shapegen = MagicMock()
        mock_shapegen.FloaterRemover = MagicMock(return_value=lambda m: m)
        mock_shapegen.DegenerateFaceRemover = MagicMock(return_value=lambda m: m)
        mock_shapegen.FaceReducer = MagicMock(return_value=lambda m, max_facenum: m)

        with patch.dict(
            "sys.modules",
            {
                "hy3dgen": MagicMock(),
                "hy3dgen.shapegen": mock_shapegen,
                "hy3dgen.rembg": MagicMock(),
                "hy3dgen.texgen": MagicMock(),
            },
        ):
            job = {
                "input": {
                    "action": "generate3d",
                    "image_base64": self.dummy_b64,
                    "texture": True,
                }
            }
            events = list(handler(job))

        stages = [e["stage"] for e in events if e.get("type") == "progress"]
        self.assertIn("texture_generation", stages)
        mock_move.assert_called_once()
        mock_offload.assert_called_once()
        mock_tex_pipe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
