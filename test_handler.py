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
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

import runpod_handler
import preload_models
from runpod_handler import (
    CHUNK_SIZE_BYTES,
    chunk_bytes,
    decode_base64_image,
    handler,
    prepare_shape_image,
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

    def test_shape_input_preserves_existing_alpha(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
        image.putpixel((4, 4), (255, 0, 0, 255))
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        decoded, _ = decode_base64_image(base64.b64encode(buf.getvalue()).decode())
        rembg = MagicMock()

        self.assertEqual(decoded.mode, "RGBA")
        self.assertIs(prepare_shape_image(decoded, rembg), decoded)
        rembg.assert_not_called()

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

    def test_shape_preload_downloads_checkpoint_without_loading_model(self):
        mock_huggingface_hub = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_huggingface_hub}), \
             patch.object(preload_models, "HUNYUAN_SHAPE_MODEL", "tencent/Hunyuan3D-2.1"), \
             patch.object(preload_models, "HUNYUAN_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-1"), \
             patch.object(preload_models, "HUNYUAN_SHAPE_USE_SAFETENSORS", False):
            preload_models.preload_shape_checkpoint()

        kwargs = mock_huggingface_hub.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "tencent/Hunyuan3D-2.1")
        self.assertIn("hunyuan3d-dit-v2-1/*.ckpt", kwargs["allow_patterns"])

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

    @patch.object(runpod_handler, "handle_generate3d")
    @patch.object(runpod_handler, "handle_custom_describe")
    @patch.object(runpod_handler, "handle_caption")
    @patch.object(runpod_handler, "handle_analyze")
    def test_handler_pipeline_runs_all_stages(
        self, mock_analyze, mock_caption, mock_describe, mock_generate3d
    ):
        mock_analyze.return_value = iter(
            [{"type": "result", "action": "analyze", "data": {"is_nsfw": False}}]
        )
        mock_caption.return_value = iter(
            [{"type": "result", "action": "caption", "data": {"caption": "a chair"}}]
        )
        mock_describe.return_value = iter(
            [{
                "type": "result",
                "action": "custom_describe",
                "data": {"description": "Ghế gỗ", "description_en": "Wooden chair"},
            }]
        )
        mock_generate3d.return_value = iter(
            [
                {"type": "file_meta", "name": "model.glb", "chunks": 1},
                {"type": "file_chunk", "index": 0, "data": "Z2xi"},
                {"type": "done"},
            ]
        )

        events = list(handler({"input": {"action": "pipeline", "image_base64": self.dummy_b64}}))

        result = next(event for event in events if event.get("action") == "pipeline")
        self.assertEqual(result["data"]["rich_prompt"], "Ghế gỗ")
        self.assertTrue(any(event.get("type") == "file_chunk" for event in events))
        mock_generate3d.assert_called_once()
        self.assertTrue(mock_generate3d.call_args.args[0]["texture"])

    @patch.object(runpod_handler, "handle_generate3d")
    @patch.object(runpod_handler, "handle_custom_describe")
    @patch.object(runpod_handler, "handle_caption")
    @patch.object(runpod_handler, "handle_analyze")
    def test_handler_pipeline_stops_on_nsfw(
        self, mock_analyze, mock_caption, mock_describe, mock_generate3d
    ):
        mock_analyze.return_value = iter(
            [{"type": "result", "action": "analyze", "data": {"is_nsfw": True}}]
        )

        events = list(handler({"input": {"action": "pipeline", "image_base64": self.dummy_b64}}))

        error = next(event for event in events if event.get("type") == "error")
        self.assertEqual(error["stage"], "moderation")
        mock_caption.assert_not_called()
        mock_describe.assert_not_called()
        mock_generate3d.assert_not_called()

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
        mock_face_reducer = MagicMock(side_effect=lambda m, max_facenum: m)
        mock_shapegen.FaceReducer = MagicMock(return_value=mock_face_reducer)

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

        shape_args = mock_shape_pipe.call_args.kwargs
        self.assertEqual(shape_args["octree_resolution"], 384)
        self.assertEqual(shape_args["num_inference_steps"], 50)
        mock_face_reducer.assert_called_once_with(mock_mesh, max_facenum=200000)

    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    @patch.object(runpod_handler.model_manager, "move_tex_pipeline")
    @patch.object(runpod_handler.model_manager, "offload_tex_pipeline")
    def test_handler_generate3d_with_texture_success(
        self, mock_offload, mock_move, mock_load_hunyuan
    ):
        """Test generate3d action with texture=True, verifying texture stage and offloading."""
        fake_glb_bytes = b"GLB_TEXTURED_DATA" * 1000
        mock_mesh = MagicMock()
        mock_mesh.visual.kind = "texture"
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

    @patch.object(runpod_handler, "HUNYUAN_USE_FLASHVDM", True)
    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    def test_handler_generate3d_fallback_on_flashvdm_empty_mesh(
        self, mock_load_hunyuan
    ):
        """Test that if FlashVDM returns None or empty mesh, fallback activates and succeeds."""
        fake_glb_bytes = b"FALLBACK_GLB_DATA" * 500
        mock_mesh = MagicMock()
        mock_mesh.export.side_effect = lambda f, file_type: f.write(fake_glb_bytes)

        mock_rembg = MagicMock(return_value=Image.new("RGB", (64, 64)))
        mock_shape_pipe = MagicMock()
        # First call (FlashVDM) returns [None], second call (fallback) returns [mock_mesh]
        mock_shape_pipe.side_effect = [[None], [mock_mesh]]

        mock_load_hunyuan.return_value = {
            "rembg": mock_rembg,
            "shape_pipeline": mock_shape_pipe,
            "tex_pipeline": None,
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

        mock_shape_pipe.enable_flashvdm.assert_any_call(enabled=False)
        self.assertEqual(mock_shape_pipe.call_count, 2)

        meta_event = next(e for e in events if e.get("type") == "file_meta")
        self.assertEqual(meta_event["name"], "model.glb")
        self.assertEqual(meta_event["size"], len(fake_glb_bytes))

    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    def test_handler_generate3d_complete_failure_yields_error_event(
        self, mock_load_hunyuan
    ):
        """Test that if both FlashVDM and fallback fail, an error event is yielded."""
        mock_rembg = MagicMock(return_value=Image.new("RGB", (64, 64)))
        mock_shape_pipe = MagicMock()
        mock_shape_pipe.side_effect = [[None], [None]]

        mock_load_hunyuan.return_value = {
            "rembg": mock_rembg,
            "shape_pipeline": mock_shape_pipe,
            "tex_pipeline": None,
        }

        mock_shapegen = MagicMock()
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

        error_events = [e for e in events if e.get("type") == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertEqual(error_events[0]["stage"], "execution")
        self.assertIn("Hunyuan3D shape generation returned empty mesh", error_events[0]["error"])

    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    def test_handler_generate3d_texture_error_is_reported(
        self, mock_load_hunyuan
    ):
        """A texture request must never silently return an untextured GLB."""
        fake_glb_bytes = b"FALLBACK_UNTEXTURED_GLB" * 500
        mock_mesh = MagicMock()
        mock_mesh.export.side_effect = lambda f, file_type: f.write(fake_glb_bytes)

        mock_rembg = MagicMock(return_value=Image.new("RGB", (64, 64)))
        mock_shape_pipe = MagicMock()
        mock_shape_pipe.return_value = [mock_mesh]

        mock_tex_pipe = MagicMock()
        mock_tex_pipe.side_effect = RuntimeError("CUDA out of memory during multiview sampling")

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
        self.assertFalse(any(e.get("type") == "file_meta" for e in events))
        error_event = next(e for e in events if e.get("type") == "error")
        self.assertIn("refusing to return an untextured model", error_event["error"])

    def test_model_manager_tex_pipeline_offload_and_move(self):
        """Move the owning pipeline once; it recursively moves its submodels."""
        mock_pipe = MagicMock()
        mock_submodel_1 = MagicMock()
        mock_submodel_2 = MagicMock()
        mock_pipe.models = {
            "delight_model": mock_submodel_1,
            "multiview_model": mock_submodel_2,
        }

        # Test move
        runpod_handler.ModelManager.move_tex_pipeline(mock_pipe, "cuda")
        mock_pipe.to.assert_called_with("cuda")
        mock_submodel_1.to.assert_not_called()
        mock_submodel_2.to.assert_not_called()

        # Test offload
        runpod_handler.ModelManager.offload_tex_pipeline(mock_pipe)
        mock_pipe.to.assert_called_with("cpu")
        mock_submodel_1.to.assert_not_called()
        mock_submodel_2.to.assert_not_called()

    def test_texture_loader_uses_repository_weight_formats(self):
        """Keep the mixed Hunyuan texture checkpoint formats explicit."""
        root = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen"
        cache_source = (root / "pipelines.py").read_text(encoding="utf-8")
        component_source = (root / "utils/multiview_utils.py").read_text(encoding="utf-8")
        unet_source = (root / "hunyuanpaint/unet/modules.py").read_text(encoding="utf-8")

        self.assertIn("os.path.join(subfolder, 'unet', 'diffusion_pytorch_model.bin')", cache_source)
        self.assertIn('f"{subfolder}/unet/diffusion_pytorch_model.safetensors"', cache_source)
        self.assertIn("torch.load(bin_path", unet_source)
        self.assertNotIn("load_file(safetensors_path", unet_source)
        self.assertIn("custom_pipeline=custom_pipeline_path", component_source)
        self.assertNotIn("**components", component_source)
        self.assertIn("UNet2p5DConditionModel.forward.__get__", component_source)
        self.assertIn("ref_scale_timing = ref_scale", unet_source)

    def test_paint_pipeline_device_matching(self):
        """Ensure texture pipeline aligns tensor devices and verifies file sizes."""
        root = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen"
        pipeline_source = (root / "hunyuanpaint/pipeline.py").read_text(encoding="utf-8")
        cache_source = (root / "pipelines.py").read_text(encoding="utf-8")
        component_source = (root / "utils/multiview_utils.py").read_text(encoding="utf-8")
        delight_source = (root / "utils/dehighlight_utils.py").read_text(encoding="utf-8")
        preload_source = (Path(__file__).parent / "preload_models.py").read_text(encoding="utf-8")

        self.assertIn("images = images.to(device=device, dtype=dtype)", pipeline_source)
        self.assertIn(".to(device)", pipeline_source)
        self.assertNotIn('.to("cuda")', pipeline_source)
        self.assertIn("os.path.getsize(os.path.join(root, path)) > 1000", cache_source)
        self.assertIn("else torch.float32", component_source)
        self.assertIn("else torch.float32", delight_source)
        self.assertIn('tex_pipeline.to("cpu")', preload_source)
        self.assertNotIn('m.pipeline.to("cpu")', preload_source)


if __name__ == "__main__":
    unittest.main()

