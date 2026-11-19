"""Submit one Runpod job, validate its streamed result, and save a GLB for review."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time

import httpx

MAX_GLB_BYTES = 50 * 1024 * 1024
# runpod_handler.py sends 512 KiB of raw GLB data per chunk.
MAX_GLB_CHUNKS = MAX_GLB_BYTES // (512 * 1024)


class StreamResult:
    def __init__(self):
        self.meta = None
        self.chunks = {}
        self.size = 0
        self.done = False
        self.analysis = None

    def add(self, event):
        if not isinstance(event, dict):
            raise ValueError("Invalid stream event")
        if event.get("error") or event.get("type") == "error":
            raise RuntimeError(event.get("error", "Worker failed"))
        kind = event.get("type")
        if kind == "file_meta":
            if (type(event.get("size")) is not int or not 0 < event["size"] <= MAX_GLB_BYTES
                    or type(event.get("chunks")) is not int or not 0 < event["chunks"] <= MAX_GLB_CHUNKS
                    or not isinstance(event.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", event["sha256"])):
                raise ValueError(f"Invalid metadata or GLB exceeds Bookaria's {MAX_GLB_BYTES // (1024 * 1024)} MiB input limit")
            if self.meta is not None and self.meta != event:
                raise ValueError("Conflicting file metadata")
            self.meta = event
        elif kind == "file_chunk":
            index = event.get("index")
            if self.meta is None or type(index) is not int or not 0 <= index < self.meta["chunks"]:
                raise ValueError("Invalid chunk index or missing metadata")
            data = event.get("data")
            if not isinstance(data, str) or len(data) > 1_000_000:
                raise ValueError("Invalid or oversized chunk")
            decoded = base64.b64decode(data, validate=True)
            if index in self.chunks:
                if self.chunks[index] != decoded:
                    raise ValueError("Conflicting duplicate chunk")
                return
            self.size += len(decoded)
            if self.size > self.meta["size"]:
                raise ValueError("Chunks exceed declared size")
            self.chunks[index] = decoded
        elif kind == "done":
            self.done = True
        elif kind == "result" and event.get("action") == "analyze":
            self.analysis = event.get("data")
        elif kind == "progress":
            print(event.get("stage"), flush=True)

    def finish(self, action):
        if action == "analyze":
            if (not isinstance(self.analysis, dict) or self.analysis.get("ok") is not True
                    or type(self.analysis.get("is_nsfw")) is not bool):
                raise ValueError("No valid analysis result received")
            return None
        if not self.done or self.meta is None or len(self.chunks) != self.meta["chunks"]:
            raise ValueError("Incomplete GLB stream")
        glb = b"".join(self.chunks[i] for i in range(self.meta["chunks"]))
        if len(glb) != self.meta["size"] or hashlib.sha256(glb).hexdigest() != self.meta["sha256"]:
            raise ValueError("GLB size or SHA-256 mismatch")
        if len(glb) < 20 or struct.unpack_from("<4sII", glb) != (b"glTF", 2, len(glb)):
            raise ValueError("Invalid GLB header")
        json_size, chunk_type = struct.unpack_from("<II", glb, 12)
        if chunk_type != 0x4E4F534A or 20 + json_size > len(glb):
            raise ValueError("Invalid GLB JSON chunk")
        model = json.loads(glb[20:20 + json_size])
        if not model.get("meshes") or not model.get("textures") or not model.get("images"):
            raise ValueError("GLB must contain mesh and texture")
        for resource in [*model.get("buffers", []), *model.get("images", [])]:
            uri = resource.get("uri")
            if uri and not uri.startswith("data:"):
                raise ValueError("GLB contains an external resource")
        return glb


def run_job(client, payload, timeout):
    # Never retry submission: a lost response may already have queued a paid job.
    response = client.post("/run", json=payload)
    response.raise_for_status()
    job = response.json()
    if not isinstance(job, dict):
        raise ValueError("Invalid Runpod submission response")
    job_id = job.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Runpod returned no valid job ID")
    print(f"Runpod job: {job_id}", flush=True)
    deadline = time.monotonic() + timeout
    result = StreamResult()
    while time.monotonic() < deadline:
        response = client.get(f"/stream/{job_id}")
        response.raise_for_status()
        state = response.json()
        if not isinstance(state, dict):
            raise ValueError("Invalid Runpod status response")
        if state.get("error") or state.get("status") in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(f"Runpod job {job_id}: {state.get('error') or state.get('status')}")
        entries = state.get("stream", [])
        if not isinstance(entries, list):
            raise ValueError("Invalid Runpod stream response")
        for entry in entries:
            if not isinstance(entry, dict) or "output" not in entry:
                raise ValueError("Invalid Runpod stream entry")
            result.add(entry["output"])
        if state.get("status") == "COMPLETED" and not entries:
            return result.finish(payload["input"]["action"])
        time.sleep(2)
    raise TimeoutError(f"Stopped waiting for {job_id}; check/cancel this job in Runpod before resubmitting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--action", choices=("analyze", "pipeline"), default="pipeline")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke-model.glb"))
    args = parser.parse_args()
    key, endpoint = os.getenv("RUNPOD_API_KEY"), os.getenv("RUNPOD_ENDPOINT_ID")
    if not key or not endpoint or not re.fullmatch(r"[A-Za-z0-9_-]+", endpoint):
        parser.error("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in the environment")
    if args.action == "pipeline" and args.output.exists():
        parser.error("Output already exists; choose a new --output path")
    with args.image.open("rb") as image:
        raw = image.read(6 * 1024 * 1024 + 1)
    if not raw or len(raw) > 6 * 1024 * 1024:
        parser.error("Image must be non-empty and at most 6 MiB")
    job_input = {"action": args.action, "image_base64": base64.b64encode(raw).decode()}
    if args.action == "pipeline":
        job_input.update(texture_resolution=1024, face_count=40000)
    payload = {"input": job_input, "policy": {"executionTimeout": 1200000, "ttl": 1800000}}
    if len(json.dumps(payload).encode()) > 9_500_000:
        parser.error("Request exceeds 9.5 MB; resize the source image")
    with httpx.Client(
        base_url=f"https://api.runpod.ai/v2/{endpoint}",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30, follow_redirects=False,
    ) as client:
        try:
            glb = run_job(client, payload, timeout=1800)
        except (httpx.HTTPError, ValueError, RuntimeError, TimeoutError) as exc:
            parser.exit(1, f"Smoke test failed: {exc}\n")
    if glb is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(glb)
        print(f"Verified GLB saved: {args.output} ({len(glb)} bytes)")
    else:
        print("Analysis smoke test passed (does not test 3D generation)")


if __name__ == "__main__":
    main()

