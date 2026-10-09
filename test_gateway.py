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
        return self.responses[url]


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
            await gateway.process_all_pipeline(image, None)

        self.assertEqual(error.exception.status_code, 415)

    async def test_process_all_runs_services_in_order_and_saves_glb(self):
        client = FakeClient(
            {
                gateway.NSFW_URL: FakeResponse({"ok": True, "is_nsfw": False}),
                gateway.CAPTION_URL: FakeResponse(
                    {"caption": "Một chiếc ghế gỗ", "original_caption": "a wooden chair"}
                ),
                gateway.CUSTOM_URL: FakeResponse({"result": "Ghế gỗ sồi, bốn chân tròn"}),
                gateway.THREED_URL: FakeResponse(content=b"glb-data"),
            }
        )

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            gateway.httpx, "AsyncClient", return_value=client
        ), patch.object(gateway, "OUTPUT_DIR", Path(output_dir)):
            result = await gateway.process_all_pipeline(upload(), None)
            model_path = Path(output_dir) / f"{result['task_id']}.glb"
            self.assertEqual(model_path.read_bytes(), b"glb-data")

        self.assertEqual(
            [call[0] for call in client.calls],
            [gateway.NSFW_URL, gateway.CAPTION_URL, gateway.CUSTOM_URL, gateway.THREED_URL],
        )
        self.assertEqual(result["results"]["model_3d"]["status"], "success")

    async def test_process_all_stops_after_nsfw_rejection(self):
        client = FakeClient({gateway.NSFW_URL: FakeResponse({"is_nsfw": True})})

        with patch.object(gateway.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(HTTPException) as error:
                await gateway.process_all_pipeline(upload(), None)

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual([call[0] for call in client.calls], [gateway.NSFW_URL])


if __name__ == "__main__":
    unittest.main()
