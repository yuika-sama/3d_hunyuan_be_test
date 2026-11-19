import base64
import hashlib
import json
import struct
import unittest
from unittest.mock import patch

import httpx
from runpod_smoke import StreamResult, run_job


def events():
    model = {"asset": {"version": "2.0"}, "meshes": [{}], "textures": [{}], "images": [{"bufferView": 0}]}
    content = json.dumps(model).encode()
    content += b" " * (-len(content) % 4)
    glb = struct.pack("<4sIIII", b"glTF", 2, 20 + len(content), len(content), 0x4E4F534A) + content
    return glb, [
        {"type": "file_meta", "size": len(glb), "chunks": 2, "sha256": hashlib.sha256(glb).hexdigest()},
        {"type": "file_chunk", "index": 0, "data": base64.b64encode(glb[:32]).decode()},
        {"type": "file_chunk", "index": 1, "data": base64.b64encode(glb[32:]).decode()},
        {"type": "done"},
    ]


class TestStreamVerification(unittest.TestCase):
    def test_reassembles_out_of_order_and_identical_repeated_chunks(self):
        glb, stream = events()
        result = StreamResult()
        for event in [stream[0], stream[2], stream[2], stream[1], stream[3]]:
            result.add(event)
        self.assertEqual(result.finish("pipeline"), glb)

    def test_rejects_incomplete_hash_mismatch_and_worker_error(self):
        _, stream = events()
        for altered in (stream[:-1], [stream[0], stream[1], stream[3]],
                        [{**stream[0], "sha256": "0" * 64}, *stream[1:]]):
            with self.subTest(altered=altered):
                result = StreamResult()
                for event in altered:
                    result.add(event)
                with self.assertRaises(ValueError):
                    result.finish("pipeline")
        with self.assertRaises(RuntimeError):
            StreamResult().add({"type": "error", "error": "texture failed"})

    def test_bookaria_accepts_raw_models_up_to_50_mib_and_100_chunks(self):
        _, stream = events()
        meta = {**stream[0], "size": 50 * 1024 * 1024, "chunks": 100}
        StreamResult().add(meta)
        for invalid in ({**meta, "size": meta["size"] + 1}, {**meta, "chunks": 101}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                StreamResult().add(invalid)

    def test_rejects_conflicting_duplicate_and_invalid_glb(self):
        _, stream = events()
        result = StreamResult()
        result.add(stream[0])
        result.add(stream[1])
        with self.assertRaises(ValueError):
            result.add({**stream[1], "data": "YWJj"})
        result = StreamResult()
        glb = b"not a glb"
        result.add({"type": "file_meta", "chunks": 1, "size": len(glb),
                    "sha256": hashlib.sha256(glb).hexdigest()})
        result.add({"type": "file_chunk", "index": 0, "data": base64.b64encode(glb).decode()})
        result.add({"type": "done"})
        with self.assertRaises(ValueError):
            result.finish("pipeline")

    def test_drain_stream_after_completed_and_submit_once(self):
        glb, stream = events()
        calls = []
        responses = iter([
            {"id": "job-123"},
            {"status": "IN_PROGRESS", "stream": [{"output": e} for e in stream[:2]]},
            {"status": "COMPLETED", "stream": [{"output": e} for e in stream[2:]]},
            {"status": "COMPLETED", "stream": []},
        ])
        def respond(request):
            calls.append((request.method, request.url.path))
            return httpx.Response(200, json=next(responses))
        with httpx.Client(base_url="https://api.runpod.ai/v2/endpoint", transport=httpx.MockTransport(respond)) as client:
            with patch("runpod_smoke.time.sleep"):
                self.assertEqual(run_job(client, {"input": {"action": "pipeline"}}, 60), glb)
        self.assertEqual(calls[0], ("POST", "/v2/endpoint/run"))
        self.assertEqual(sum(method == "POST" for method, _ in calls), 1)

    def test_completed_without_analysis_and_provider_failure_are_errors(self):
        for status in ("FAILED", "COMPLETED"):
            responses = iter([{"id": "job-123"}, {"status": status, "stream": []}])
            with httpx.Client(base_url="https://api.runpod.ai/v2/endpoint",
                              transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(responses)))) as client:
                with self.assertRaises((ValueError, RuntimeError)):
                    run_job(client, {"input": {"action": "analyze"}}, 60)

    def test_malformed_provider_responses_fail_cleanly(self):
        for response in ([], {"status": "COMPLETED", "stream": [None]},
                         {"status": "COMPLETED", "stream": "invalid"}):
            responses = iter([{"id": "job-123"}, response])
            with self.subTest(response=response), httpx.Client(
                base_url="https://api.runpod.ai/v2/endpoint",
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(responses))),
            ) as client:
                with self.assertRaises(ValueError):
                    run_job(client, {"input": {"action": "analyze"}}, 60)

    def test_untextured_glb_fails_even_with_correct_checksum(self):
        model = json.dumps({"meshes": [{}]}).encode()
        model += b" " * (-len(model) % 4)
        glb = struct.pack("<4sIIII", b"glTF", 2, 20 + len(model), len(model), 0x4E4F534A) + model
        result = StreamResult()
        result.add({"type": "file_meta", "size": len(glb), "chunks": 1,
                    "sha256": hashlib.sha256(glb).hexdigest()})
        result.add({"type": "file_chunk", "index": 0, "data": base64.b64encode(glb).decode()})
        result.add({"type": "done"})
        with self.assertRaisesRegex(ValueError, "texture"):
            result.finish("pipeline")

    def test_submission_failure_is_never_retried(self):
        calls = []
        def respond(request):
            calls.append(request)
            raise httpx.ReadTimeout("unknown submission state", request=request)
        with httpx.Client(base_url="https://api.runpod.ai/v2/endpoint", transport=httpx.MockTransport(respond)) as client:
            with self.assertRaises(httpx.ReadTimeout):
                run_job(client, {"input": {"action": "pipeline"}}, 60)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

