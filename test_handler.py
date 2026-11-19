"""
Unit and Integration Tests for Runpod Serverless Worker
Covers:
- Image decoding and validation (base64, data URIs, error cases)
- Chunking and SHA256 GLB reassembly
- Handler action routing, error reporting, and stream generator events
- Model lifecycle and lean container configuration
"""

import base64
import hashlib
import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_container_excludes_blip_and_llava(self):
        root = Path(__file__).parent
        build_source = (root / "Dockerfile").read_text(encoding="utf-8")
        handler_source = (root / "runpod_handler.py").read_text(encoding="utf-8")
        requirements = (root / "requirements-runpod.txt").read_text(encoding="utf-8")

        for removed in ("ollama", "llava", "BLIP_MODEL_NAME", "handle_caption", "handle_custom_describe"):
            self.assertNotIn(removed, build_source + handler_source)
        self.assertNotIn("deep-translator", requirements)
        self.assertNotIn("pytest", requirements)

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
    @patch.object(runpod_handler, "handle_analyze")
    def test_handler_pipeline_runs_all_stages(self, mock_analyze, mock_generate3d):
        mock_analyze.side_effect = [
            iter([{"type": "result", "action": "analyze", "data": {"is_nsfw": False}}]),
            iter([{"type": "result", "action": "analyze", "data": {"is_nsfw": False}}]),
        ]
        mock_generate3d.return_value = iter(
            [
                {"type": "file_meta", "name": "model.glb", "chunks": 1},
                {"type": "file_chunk", "index": 0, "data": "Z2xi"},
                {"type": "done"},
            ]
        )

        events = list(handler({"input": {
            "action": "pipeline",
            "image_base64": self.dummy_b64,
            "texture_image_base64": self.dummy_b64,
        }}))

        result = next(event for event in events if event.get("action") == "pipeline")
        self.assertFalse(result["data"]["moderation"]["is_nsfw"])
        self.assertFalse(result["data"]["texture_moderation"]["is_nsfw"])
        self.assertTrue(any(event.get("type") == "file_chunk" for event in events))
        self.assertEqual(mock_analyze.call_count, 2)
        mock_generate3d.assert_called_once()
        self.assertTrue(mock_generate3d.call_args.args[0]["texture"])

    @patch.object(runpod_handler, "handle_generate3d")
    @patch.object(runpod_handler, "handle_analyze")
    def test_handler_pipeline_stops_on_nsfw(self, mock_analyze, mock_generate3d):
        mock_analyze.return_value = iter(
            [{"type": "result", "action": "analyze", "data": {"is_nsfw": True}}]
        )

        events = list(handler({"input": {"action": "pipeline", "image_base64": self.dummy_b64}}))

        error = next(event for event in events if event.get("type") == "error")
        self.assertEqual(error["stage"], "moderation")
        mock_generate3d.assert_not_called()

    @patch.object(runpod_handler, "handle_generate3d")
    @patch.object(runpod_handler, "handle_analyze")
    def test_handler_pipeline_stops_on_texture_nsfw(self, mock_analyze, mock_generate3d):
        mock_analyze.side_effect = [
            iter([{"type": "result", "action": "analyze", "data": {"is_nsfw": False}}]),
            iter([{"type": "result", "action": "analyze", "data": {"is_nsfw": True}}]),
        ]

        events = list(handler({"input": {
            "action": "pipeline",
            "image_base64": self.dummy_b64,
            "texture_image_base64": self.dummy_b64,
        }}))

        error = next(event for event in events if event.get("type") == "error")
        self.assertEqual(error["stage"], "texture_moderation")
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

        mock_rembg = MagicMock(side_effect=lambda image: image)
        mock_shape_pipe = MagicMock()
        mock_shape_pipe.return_value = [mock_mesh]

        mock_load_hunyuan.return_value = {
            "rembg": mock_rembg,
            "shape_pipeline": mock_shape_pipe,
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
        self.assertNotIn("mc_algo", shape_args)
        mock_shapegen.FloaterRemover.assert_called_once()
        mock_face_reducer.assert_called_once_with(mock_mesh, max_facenum=40000)

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

        mock_rembg = MagicMock(side_effect=lambda image: image)
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
                    "texture_image_base64": create_dummy_image_b64(color=(0, 0, 255)),
                    "texture": True,
                }
            }
            events = list(handler(job))

        stages = [e["stage"] for e in events if e.get("type") == "progress"]
        self.assertIn("texture_generation", stages)
        mock_load_hunyuan.assert_called_once_with(
            load_texture=True, texture_subfolder=runpod_handler.HUNYUAN_TEX_SUBFOLDER,
        )
        mock_move.assert_called_once()
        mock_offload.assert_called_once()
        mock_tex_pipe.assert_called_once()
        self.assertEqual(
            mock_tex_pipe.call_args.args[1].getpixel((0, 0))[:3],
            (0, 0, 255),
        )

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

    @patch.object(runpod_handler, "HUNYUAN_USE_FLASHVDM", False)
    @patch.object(runpod_handler.model_manager, "load_hunyuan")
    def test_handler_generate3d_complete_failure_yields_error_event(
        self, mock_load_hunyuan
    ):
        """Standard decoding runs once and preserves the reason for a failure."""
        mock_rembg = MagicMock(return_value=Image.new("RGB", (64, 64)))
        mock_shape_pipe = MagicMock()
        mock_shape_pipe.side_effect = [[None], RuntimeError("out of memory")]

        mock_load_hunyuan.return_value = {
            "rembg": mock_rembg,
            "shape_pipeline": mock_shape_pipe,
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
            mock_shape_pipe.assert_called_once()
            self.assertIn("empty mesh", events[-1]["error"])
            events = list(handler(job))
            self.assertEqual(mock_shape_pipe.call_count, 2)
            self.assertEqual(events[-1]["stage"], "execution")
            self.assertEqual(events[-1]["error"], "out of memory")

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
                    "texture_image_base64": self.dummy_b64,
                    "texture": True,
                }
            }
            events = list(handler(job))

        stages = [e["stage"] for e in events if e.get("type") == "progress"]
        self.assertIn("texture_generation", stages)
        self.assertFalse(any(e.get("type") == "file_meta" for e in events))
        error_event = next(e for e in events if e.get("type") == "error")
        self.assertIn("refusing to return an untextured model", error_event["error"])

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
        self.assertIn("Basic2p5DTransformerBlock.forward.__get__", component_source)
        self.assertIn("ref_scale_timing = ref_scale", unet_source)
        self.assertIn("original_inference_steps=30", component_source)
        self.assertIn('cross_attention_kwargs.pop("position_attn_mask"', unet_source)
        self.assertNotIn("position_voxel_indices", unet_source)

    def test_paint_options_follow_each_request(self):
        from importlib.util import module_from_spec, spec_from_file_location

        root = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen"
        def load(name, path, dependencies):
            spec = spec_from_file_location("hy3dgen.texgen." + name, root / path)
            module = module_from_spec(spec)
            with patch.dict("sys.modules", {dep: MagicMock() for dep in dependencies}):
                spec.loader.exec_module(module)
            return module

        mv = load("utils.multiview_utils", "utils/multiview_utils.py", [
            "diffusers", "hy3dgen.texgen.hunyuanpaint.unet.modules",
        ])
        net = mv.Multiview_Diffusion_Net.__new__(mv.Multiview_Diffusion_Net)
        net.device = "cpu"
        net.view_size = 8
        net.pipeline = MagicMock(device="cpu")
        image = Image.new("RGB", (8, 8))
        for count in (1, 2, 6):
            net([image] * count, [image, image], [0])
            self.assertEqual(net.pipeline.call_args.kwargs["camera_info_ref"], [[0] * count])

        paint = load("pipelines", "pipelines.py", ["hy3dgen.texgen." + name for name in (
            "differentiable_renderer.mesh_render", "utils.dehighlight_utils",
            "utils.multiview_utils", "utils.imagesuper_utils", "utils.uv_warp_utils",
        )])
        config = paint.Hunyuan3DTexGenConfig("", "", "hunyuan3d-paint-v2-0", enable_delight=False)
        pipeline = paint.Hunyuan3DPaintPipeline(config)
        paint.mesh_uv_wrap.side_effect = RuntimeError("reached UV stage")
        for flag, calls in [(None, 0), (True, 1), (None, 1), (True, 2)]:
            with self.assertRaisesRegex(RuntimeError, "reached UV stage"):
                pipeline(None, image, enable_delight=flag)
            self.assertEqual(paint.Light_Shadow_Remover.return_value.call_count, calls)
        paint.Light_Shadow_Remover.assert_called_once_with(config)

    def test_paint_pipeline_device_matching(self):
        """Ensure texture pipeline aligns tensor devices and verifies file sizes."""
        root = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen"
        pipeline_source = (root / "hunyuanpaint/pipeline.py").read_text(encoding="utf-8")
        cache_source = (root / "pipelines.py").read_text(encoding="utf-8")
        component_source = (root / "utils/multiview_utils.py").read_text(encoding="utf-8")
        delight_source = (root / "utils/dehighlight_utils.py").read_text(encoding="utf-8")

        self.assertIn("images = images.to(device=device, dtype=dtype)", pipeline_source)
        self.assertIn(".to(device)", pipeline_source)
        self.assertNotIn('.to("cuda")', pipeline_source)
        self.assertIn("for module in self.unet.modules()", pipeline_source)
        self.assertIn("self.scheduler.set_timesteps(num_inference_steps=num_inference_steps", pipeline_source)
        self.assertNotIn("class DDIMSolver", pipeline_source)
        self.assertIn("os.path.getsize(os.path.join(root, path)) > 1000", cache_source)
        self.assertIn("else torch.float32", component_source)
        self.assertIn("else torch.float32", delight_source)
        preload_source = (Path(__file__).parent / "preload_models.py").read_text(encoding="utf-8")
        self.assertIn('tex_pipeline.to("cpu")', preload_source)
        self.assertNotIn('m.pipeline.to("cpu")', preload_source)

    def test_texture_delight_disabled_by_default_and_preserves_colors(self):
        """Verify delight model is bypassed by default to preserve RGBA and true colors."""
        root = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/hy3dgen/texgen"
        pipelines_source = (root / "pipelines.py").read_text(encoding="utf-8")
        paint_source = (root / "hunyuanpaint/pipeline.py").read_text(encoding="utf-8")
        delight_source = (root / "utils/dehighlight_utils.py").read_text(encoding="utf-8")
        handler_source = (Path(__file__).parent / "runpod_handler.py").read_text(encoding="utf-8")

        self.assertIn("enable_delight=None", pipelines_source)
        self.assertIn("getattr(self.config, 'enable_delight', False)", pipelines_source)
        self.assertIn("HUNYUAN_ENABLE_DELIGHT", handler_source)
        self.assertIn("numpy.full((rgba.size[1], rgba.size[0], 3), 127", paint_source)
        self.assertIn("mode='RGBA'", delight_source)


class TestInputSafety(unittest.TestCase):
    def setUp(self):
        self.input = {"action": "pipeline", "image": create_dummy_image_b64()}

    def test_rejects_resource_limits_and_ambiguous_inputs_before_loading_models(self):
        invalid = [
            {"texture": "false"}, {"enable_delight": 1}, {"num_inference_steps": 1000},
            {"face_count": -1}, {"seed": True}, {"octree_resolution": 1024},
            {"texture_resolution": 4096}, {"texture_size": 0},
            {"guidance_scale": float("nan")}, {"guidance_scale": float("inf")},
            {"guidance_scale": 10**1000},
            {"texture_images": []}, {"texture_images": ["x"] * 7},
            {"texture_images": "not-a-list"}, {"texture_images": [None]},
            {"texture_image": ""}, {"texture_image": "x", "texture_images": ["y"]},
            {"image_base64": "different"}, {"texture_subfolder": "../other"},
            {"unknown": 123},
        ]
        with patch.object(runpod_handler.model_manager, "load_hunyuan") as load:
            for fields in invalid:
                with self.subTest(fields=fields):
                    events = list(handler({"input": {**self.input, **fields}}))
                    self.assertEqual(events[0]["stage"], "validate_input")
                    self.assertEqual(events[0]["type"], "error")
            load.assert_not_called()

    def test_aliases_and_total_encoded_payload(self):
        data = runpod_handler.validate_input({**self.input, "texture_size": 1024, "texture": False})
        self.assertEqual(data["texture_resolution"], 1024)
        self.assertFalse(data["texture"])
        self.assertNotIn("image", data)
        with patch.object(runpod_handler, "MAX_INPUT_BYTES", 250):
            with self.assertRaises(ValueError):
                runpod_handler.validate_input({**self.input, "texture_images": [self.input["image"]] * 3})

    def test_image_size_pixels_and_strict_base64(self):
        with patch.object(runpod_handler, "MAX_IMAGE_SIZE_BYTES", 4):
            with self.assertRaises(ValueError):
                decode_base64_image(self.input["image"])
        with patch.object(runpod_handler, "MAX_IMAGE_PIXELS", 32):
            with self.assertRaises(ValueError):
                decode_base64_image(self.input["image"])
        with self.assertRaises(ValueError):
            decode_base64_image(self.input["image"] + "!!!")

    def test_every_multiview_texture_is_moderated_and_block_stops_generation(self):
        for alias in ("texture_images", "texture_images_base64"):
            with self.subTest(alias=alias), patch.object(runpod_handler, "handle_analyze") as analyze, \
                    patch.object(runpod_handler, "handle_generate3d") as generate:
                analyze.side_effect = [iter([{"type": "result", "data": {"is_nsfw": value}}])
                                       for value in (False, False, True)]
                events = list(handler({"input": {**self.input, alias: ["safe", "blocked"]}}))
                self.assertEqual(analyze.call_count, 3)
                self.assertEqual(analyze.call_args.args[0]["image_base64"], "blocked")
                self.assertEqual(events[-1]["stage"], "texture_moderation")
                self.assertEqual(events[-1]["image_index"], 1)
                generate.assert_not_called()

    def test_malformed_moderation_fails_closed(self):
        with patch.object(runpod_handler, "handle_analyze", return_value=iter([
            {"type": "result", "data": {"ok": True}},
        ])), patch.object(runpod_handler, "handle_generate3d") as generate:
            events = list(handler({"input": self.input}))
            self.assertEqual(events[-1]["type"], "error")
            generate.assert_not_called()

    def test_selected_texture_checkpoint_reaches_loader(self):
        manager = runpod_handler.ModelManager()
        with patch.object(manager, "load_shape_pipeline", return_value=(None, None)), \
                patch.object(manager, "load_tex_pipeline") as load:
            manager.load_hunyuan(texture_subfolder="hunyuan3d-paint-v2-0-turbo")
            load.assert_called_once_with(subfolder="hunyuan3d-paint-v2-0-turbo")

    def test_preload_failure_stops_startup(self):
        rembg = MagicMock()
        rembg.BackgroundRemover.side_effect = RuntimeError("download failed")
        with patch.dict("sys.modules", {"hy3dgen": MagicMock(), "hy3dgen.rembg": rembg}), \
                patch.object(preload_models, "preload_shape_checkpoint"):
            with self.assertRaisesRegex(RuntimeError, "BackgroundRemover"):
                preload_models.preload()

    def test_preload_uses_configured_texture_checkpoint(self):
        import os
        texgen = MagicMock()
        with patch.dict("sys.modules", {"hy3dgen": MagicMock(), "hy3dgen.rembg": MagicMock(),
                                        "hy3dgen.texgen": texgen}), \
                patch.object(preload_models, "preload_shape_checkpoint"), \
                patch.dict(os.environ, {"HUNYUAN_TEX_MODEL": "test/model",
                                        "HUNYUAN_TEX_SUBFOLDER": "hunyuan3d-paint-v2-0-turbo"}):
            preload_models.preload(full_texture=True)
        texgen.Hunyuan3DPaintPipeline.from_pretrained.assert_called_once_with(
            "test/model", subfolder="hunyuan3d-paint-v2-0-turbo",
        )

    def test_closing_shape_generator_offloads_pipeline(self):
        shape = MagicMock()
        with patch.dict("sys.modules", {"hy3dgen": MagicMock(), "hy3dgen.shapegen": MagicMock()}), \
                patch.object(runpod_handler.model_manager, "load_hunyuan", return_value={
                    "rembg": lambda image: image, "shape_pipeline": shape,
                }):
            generator = handler({"input": {**self.input, "action": "generate3d", "texture": False}})
            for event in generator:
                if event.get("stage") == "shape_generation":
                    break
            shape.to.reset_mock()
            generator.close()
            shape.to.assert_called_once_with("cpu")


class TestLocalTextureFailure(unittest.TestCase):
    def test_local_worker_does_not_export_missing_or_untextured_paint(self):
        # Extract only the existing method so this regression needs no CUDA/model imports.
        import ast
        import types
        source = Path(__file__).parent / "3d_generative/Hunyuan3D-2-main/api_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelWorker")
        method = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "_generate")
        method.decorator_list = []
        mesh = MagicMock()
        mesh.visual.kind = "vertex"
        scope = {"base64": base64, "BytesIO": io.BytesIO, "trimesh": MagicMock(),
                 "FloaterRemover": lambda: lambda mesh: mesh,
                 "DegenerateFaceRemover": lambda: lambda mesh: mesh,
                 "FaceReducer": lambda: lambda mesh, **kwargs: mesh,
                 "load_image_from_base64": lambda image: image, "torch": MagicMock()}
        scope["trimesh"].load.return_value = mesh
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        for paint in (None, MagicMock(return_value=mesh)):
            with self.subTest(paint=paint):
                instance = types.SimpleNamespace(rembg=lambda image: image, pipeline_tex=paint, device="cpu")
                with self.assertRaisesRegex(RuntimeError, "[Tt]exture"):
                    scope["_generate"](instance, "test", {"image": "image", "mesh": "eA==", "texture": True})
        mesh.export.assert_not_called()


class TestRunpodSDK(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_sends_error_instead_of_success_for_handler_error_event(self):
        from runpod.serverless.modules import rp_job
        with patch.object(rp_job, "send_result", new_callable=AsyncMock) as send, \
                patch.object(rp_job, "stream_result", new_callable=AsyncMock):
            result = await rp_job.handle_job(None, {"handler": handler}, {"id": "test", "input": {}})
        self.assertIn("error", result)
        self.assertNotIn("output", result)
        self.assertEqual(send.call_args.args[1], result)


if __name__ == "__main__":
    unittest.main()

