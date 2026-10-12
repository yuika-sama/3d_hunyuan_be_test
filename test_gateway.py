import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

import gateway


class FakeResponse:
    def __init__(self, data=None, content=b"", status_code=200):
        self.data = data
        self.content = content
        self.status_code = status_code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise gateway.httpx.HTTPStatusError("backend error", request=MagicMock(), response=MagicMock())


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses[url]
        return response.pop(0) if isinstance(response, list) else response


def upload(content=b"image-bytes"):
    image = MagicMock()
    image.filename = "chair.png"
    image.content_type = "image/png"
    image.read = AsyncMock(return_value=content)
    return image


class TestUnifiedPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_process_all_rejects_unsupported_image(self):
        image = upload()
        image.content_type = "image/svg+xml"

        with self.assertRaises(HTTPException) as error:
            await gateway.process_all_pipeline(image, upload())

        self.assertEqual(error.exception.status_code, 415)

    async def test_process_all_runs_services_in_order_and_saves_glb(self):
        client = FakeClient(
            {
                gateway.NSFW_URL: FakeResponse({"ok": True, "is_nsfw": False}),
                gateway.THREED_URL: FakeResponse(content=b"glb-data"),
            }
        )

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            gateway.httpx, "AsyncClient", return_value=client
        ), patch.object(gateway, "OUTPUT_DIR", Path(output_dir)):
            result = await gateway.process_all_pipeline(upload(b"shape"), upload(b"texture"))
            model_path = Path(output_dir) / f"{result['task_id']}.glb"
            self.assertEqual(model_path.read_bytes(), b"glb-data")

        self.assertEqual(
            [call[0] for call in client.calls],
            [gateway.NSFW_URL, gateway.NSFW_URL, gateway.THREED_URL],
        )
        self.assertEqual(result["results"]["model_3d"]["status"], "success")
        model_payload = client.calls[-1][1]["json"]
        self.assertEqual(model_payload["image"], "c2hhcGU=")
        self.assertEqual(model_payload["texture_image"], "dGV4dHVyZQ==")

    async def test_process_all_stops_after_nsfw_rejection(self):
        client = FakeClient({gateway.NSFW_URL: FakeResponse({"is_nsfw": True})})

        with patch.object(gateway.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(HTTPException) as error:
                await gateway.process_all_pipeline(upload(), upload())

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual([call[0] for call in client.calls], [gateway.NSFW_URL])

    async def test_process_all_stops_after_texture_nsfw_rejection(self):
        client = FakeClient({
            gateway.NSFW_URL: [
                FakeResponse({"is_nsfw": False}),
                FakeResponse({"is_nsfw": True}),
            ]
        })

        with patch.object(gateway.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(HTTPException) as error:
                await gateway.process_all_pipeline(upload(), upload())

        self.assertEqual(error.exception.detail["stage"], "texture_moderation")
        self.assertEqual(
            [call[0] for call in client.calls],
            [gateway.NSFW_URL, gateway.NSFW_URL],
        )

    async def test_generate3d_sends_shape_and_texture_separately(self):
        client = FakeClient({gateway.THREED_URL: FakeResponse(content=b"glb-data")})

        with patch.object(gateway.httpx, "AsyncClient", return_value=client):
            response = await gateway.generate3d(upload(b"shape"), upload(b"texture"))

        self.assertEqual(response.body, b"glb-data")
        payload = client.calls[0][1]["json"]
        self.assertEqual(payload["image"], "c2hhcGU=")
        self.assertEqual(payload["texture_image"], "dGV4dHVyZQ==")

    async def test_process_all_single_image_uses_same_image_for_shape_and_texture(self):
        client = FakeClient(
            {
                gateway.NSFW_URL: FakeResponse({"ok": True, "is_nsfw": False}),
                gateway.THREED_URL: FakeResponse(content=b"glb-data"),
            }
        )

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            gateway.httpx, "AsyncClient", return_value=client
        ), patch.object(gateway, "OUTPUT_DIR", Path(output_dir)):
            result = await gateway.process_all_pipeline(upload(b"single-image"))
            model_path = Path(output_dir) / f"{result['task_id']}.glb"
            self.assertEqual(model_path.read_bytes(), b"glb-data")

        # Only one NSFW check is done when single image is provided
        self.assertEqual(
            [call[0] for call in client.calls],
            [gateway.NSFW_URL, gateway.THREED_URL],
        )
        self.assertEqual(result["results"]["model_3d"]["status"], "success")
        model_payload = client.calls[-1][1]["json"]
        self.assertEqual(model_payload["image"], "c2luZ2xlLWltYWdl")
        self.assertEqual(model_payload["texture_image"], "c2luZ2xlLWltYWdl")

    async def test_generate3d_single_image_uses_same_image_for_texture(self):
        client = FakeClient({gateway.THREED_URL: FakeResponse(content=b"glb-data")})

        with patch.object(gateway.httpx, "AsyncClient", return_value=client):
            response = await gateway.generate3d(upload(b"single-image"))

        self.assertEqual(response.body, b"glb-data")
        payload = client.calls[0][1]["json"]
        self.assertEqual(payload["image"], "c2luZ2xlLWltYWdl")
        self.assertEqual(payload["texture_image"], "c2luZ2xlLWltYWdl")


if __name__ == "__main__":
    unittest.main()
