import base64
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import gateway


def png(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


IMAGE = png()
TEXTURE = png("blue")
GLB = struct.pack("<4sII", b"glTF", 2, 12)


def upload(data=IMAGE):
    result = MagicMock(filename="image.png", content_type="image/png")
    result.read = AsyncMock(return_value=data)
    return result


class FakeResponse:
    def __init__(self, data=None, content=b"", status_code=200):
        self.data, self.content, self.status_code = data, content, status_code

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


def client(moderation=None, glb=GLB):
    return FakeClient({
        gateway.NSFW_URL: moderation or FakeResponse({"ok": True, "is_nsfw": False}),
        gateway.THREED_URL: FakeResponse(content=glb),
    })


class TestUnifiedPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_both_routes_moderate_source_and_texture(self):
        for route in (gateway.generate3d, gateway.process_all_pipeline):
            with self.subTest(route=route.__name__), tempfile.TemporaryDirectory() as output:
                fake = client()
                with patch.object(gateway.httpx, "AsyncClient", return_value=fake), \
                        patch.object(gateway, "OUTPUT_DIR", Path(output)):
                    result = await route(upload(), upload(TEXTURE))
                self.assertEqual([call[0] for call in fake.calls],
                                 [gateway.NSFW_URL, gateway.NSFW_URL, gateway.THREED_URL])
                payload = fake.calls[-1][1]["json"]
                self.assertEqual(base64.b64decode(payload["image"]), IMAGE)
                self.assertEqual(base64.b64decode(payload["texture_image"]), TEXTURE)
                if route == gateway.generate3d:
                    self.assertEqual(result.body, GLB)
                else:
                    self.assertEqual((Path(output) / f"{result['task_id']}.glb").read_bytes(), GLB)

    async def test_single_image_reuses_source_for_texture(self):
        fake = client()
        with patch.object(gateway.httpx, "AsyncClient", return_value=fake):
            response = await gateway.generate3d(upload())
        self.assertEqual(response.body, GLB)
        self.assertEqual([call[0] for call in fake.calls], [gateway.NSFW_URL, gateway.THREED_URL])
        self.assertEqual(fake.calls[-1][1]["json"]["image"], fake.calls[-1][1]["json"]["texture_image"])

    async def test_both_routes_stop_on_blocked_source_or_texture(self):
        for route in (gateway.generate3d, gateway.process_all_pipeline):
            for blocked_index in (0, 1):
                responses = [FakeResponse({"ok": True, "is_nsfw": i == blocked_index}) for i in range(2)]
                fake = client(responses)
                with self.subTest(route=route.__name__, index=blocked_index), \
                        patch.object(gateway.httpx, "AsyncClient", return_value=fake):
                    with self.assertRaises(HTTPException) as raised:
                        await route(upload(), upload(TEXTURE))
                self.assertEqual(raised.exception.status_code, 400)
                self.assertEqual(len(fake.calls), blocked_index + 1)
                self.assertTrue(all(url == gateway.NSFW_URL for url, _ in fake.calls))

    async def test_invalid_moderation_is_not_treated_as_safe(self):
        for data in ({}, {"ok": False}, {"ok": True, "is_nsfw": "false"}, [], None):
            fake = client(FakeResponse(data))
            with self.subTest(data=data), patch.object(gateway.httpx, "AsyncClient", return_value=fake):
                with self.assertRaises(HTTPException) as raised:
                    await gateway.generate3d(upload())
            self.assertEqual(raised.exception.status_code, 502)
            self.assertEqual(len(fake.calls), 1)

    async def test_upload_validation_and_bounded_read(self):
        for data, code in ((b"", 400), (b"not-an-image", 400), (b"x" * 101, 413)):
            image = upload(data)
            with self.subTest(code=code), patch.object(gateway, "MAX_IMAGE_BYTES", 100):
                with self.assertRaises(HTTPException) as raised:
                    await gateway.read_image(image)
            self.assertEqual(raised.exception.status_code, code)
            image.read.assert_awaited_once_with(101)
        image = upload()
        image.content_type = "image/svg+xml"
        with self.assertRaises(HTTPException) as raised:
            await gateway.read_image(image)
        self.assertEqual(raised.exception.status_code, 415)
        with patch.object(gateway, "MAX_IMAGE_PIXELS", 1):
            with self.assertRaises(HTTPException):
                await gateway.read_image(upload())

    async def test_invalid_glb_not_saved(self):
        with tempfile.TemporaryDirectory() as output:
            with patch.object(gateway, "OUTPUT_DIR", Path(output)), \
                    patch.object(gateway.httpx, "AsyncClient", return_value=client(glb=b"error-page")):
                with self.assertRaises(HTTPException) as raised:
                    await gateway.process_all_pipeline(upload())
            self.assertEqual(raised.exception.status_code, 502)
            self.assertEqual(list(Path(output).iterdir()), [])


class TestAuthentication(unittest.TestCase):
    def test_generation_and_download_require_server_key(self):
        with TestClient(gateway.app) as http, patch.dict(os.environ, {"GATEWAY_API_KEY": "test-key"}):
            self.assertEqual(http.get("/health").status_code, 200)
            for path in ("/analyze", "/generate3d", "/pipeline/process-all"):
                response = http.post(path, files={"image": ("image.png", IMAGE, "image/png")})
                self.assertEqual(response.status_code, 401)
            path = "/pipeline/result/00000000-0000-0000-0000-000000000001.glb"
            self.assertEqual(http.get(path).status_code, 401)
            self.assertEqual(http.get(path, headers={"Authorization": "Bearer test-key"}).status_code, 404)
        with TestClient(gateway.app) as http, patch.dict(os.environ, {"GATEWAY_API_KEY": ""}):
            self.assertEqual(http.get(path).status_code, 503)


if __name__ == "__main__":
    unittest.main()
