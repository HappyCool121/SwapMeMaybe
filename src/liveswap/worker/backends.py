from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from liveswap.archive import decode_image
from liveswap.contracts import Crop, Profile, Window, frame_pts
from liveswap.worker.graph import build_graph


class MockBackend:
    """No clothes replacement: exercises transport, buffering, and interpolation only."""

    def __init__(self, profile: Profile, delay: float = 0):
        self.profile, self.delay = profile, delay

    async def warm(self):
        pass

    async def generate(self, window, frames, masks, garment):
        await asyncio.sleep(self.delay)
        output = []
        for index, frame in enumerate(frames):
            output.append(frame)
            if index < len(frames) - 1:
                output.append(((frame.astype(np.uint16) + frames[index + 1].astype(np.uint16)) // 2).astype(np.uint8))
        return output

    async def cancel(self):
        pass

    async def close(self):
        pass


class Rife:
    def __init__(self, repo: Path, weights: Path):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("RIFE production backend requires an NVIDIA CUDA GPU")
        if not (weights / "flownet.pkl").is_file():
            raise FileNotFoundError("RIFE weights missing: expected RIFE_WEIGHTS/flownet.pkl")
        sys.path.insert(0, str(repo.resolve()))
        module = importlib.import_module("model.RIFE")
        self.model = module.Model()
        state = torch.load(weights / "flownet.pkl", map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {name.removeprefix("module."): value for name, value in state.items()}
        self.model.flownet.load_state_dict(state, strict=True)
        self.model.eval()
        self.torch = torch
        self.weights_sha256 = hashlib.sha256((weights / "flownet.pkl").read_bytes()).hexdigest()

    def interpolate(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        height, width = frames[0].shape[:2]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        tensors = [torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32) / 255 for frame in frames]
        output = []
        with torch.inference_mode():
            for index, frame in enumerate(frames):
                output.append(frame)
                if index + 1 < len(frames):
                    first = torch.nn.functional.pad(tensors[index], (0, pad_w, 0, pad_h))
                    second = torch.nn.functional.pad(tensors[index + 1], (0, pad_w, 0, pad_h))
                    middle = self.model.inference(first, second, timestep=0.5)
                    middle = middle[0, :, :height, :width].permute(1, 2, 0).clamp(0, 1)
                    output.append((middle * 255).round().byte().cpu().numpy())
        return output


class ComfyBackend:
    def __init__(self, profile: Profile, data_dir: Path, url: str, rife_repo: Path, rife_weights: Path):
        self.profile, self.data_dir = profile, data_dir
        self.url = url.rstrip("/")
        self.rife_repo, self.rife_weights = rife_repo, rife_weights
        self.http = httpx.AsyncClient(timeout=30)
        self.rife = None
        self.unhealthy = False

    async def warm(self):
        deadline = time.monotonic() + 600
        while True:
            try:
                response = await self.http.get(self.url + "/object_info")
                response.raise_for_status()
                info = response.json()
                for node in ("LiveSwapLoadWindow", "LiveSwapSaveFrames", "WanVaceToVideo", "TrimVideoLatent"):
                    if node not in info:
                        raise RuntimeError(f"required ComfyUI node is missing: {node}")
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("ComfyUI did not start") from None
                await asyncio.sleep(1)
        self.rife = await asyncio.to_thread(Rife, self.rife_repo, self.rife_weights)
        profile = self.profile
        dummy = np.full((profile.height, profile.width, 3), 127, np.uint8)
        garment = self.data_dir / "assets" / ("0" * 64 + ".png")
        Image.fromarray(dummy).save(garment)
        window = Window(session_id="0" * 32, window_id=0, profile_id=profile.id,
                        garment_hash="0" * 64, frame_ids=list(range(0, profile.length * 2, 2)),
                        pts_us=[frame_pts(i) for i in range(0, profile.length * 2, 2)],
                        crop=Crop(x=0, y=0, width=profile.width, height=profile.height), budget_ms=120_000)
        # Includes a real diffusion + VAE + RIFE invocation; /readyz stays false until it finishes.
        await self.generate(window, [dummy] * profile.length, [np.full(dummy.shape[:2], 255, np.uint8)] * profile.length, garment)
        garment.unlink(missing_ok=True)

    async def generate(self, window: Window, frames, masks, garment: Path):
        job_dir = self.data_dir / "jobs" / window.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
                Image.fromarray(frame).save(job_dir / f"frame-{index:03}.png")
                Image.fromarray(mask).save(job_dir / f"mask-{index:03}.png")
            (job_dir / "input.json").write_text(json.dumps({"count": len(frames)}))
            graph = build_graph(self.profile, window.job_id, window.garment_hash)
            response = await self.http.post(self.url + "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
            response.raise_for_status()
            submitted = response.json()
            if "prompt_id" not in submitted:
                raise RuntimeError(f"ComfyUI rejected the graph: {submitted.get('error', 'missing prompt_id')}")
            prompt_id = submitted["prompt_id"]
            started = time.monotonic()
            while time.monotonic() - started < 900:
                response = await self.http.get(self.url + f"/history/{prompt_id}")
                response.raise_for_status()
                record = response.json().get(prompt_id)
                if record:
                    if record.get("status", {}).get("status_str") == "error":
                        raise RuntimeError("ComfyUI inference failed; inspect the private worker log")
                    images = record.get("outputs", {}).get("12", {}).get("images", [])
                    if not images:
                        raise RuntimeError("ComfyUI finished without garment frames")
                    generated = []
                    for image in images:
                        response = await self.http.get(self.url + "/view", params=image)
                        response.raise_for_status()
                        generated.append(decode_image(response.content, size=(self.profile.width, self.profile.height)))
                    if len(generated) != self.profile.length:
                        raise RuntimeError("ComfyUI returned an incorrect frame count")
                    # A cancelled asyncio thread cannot stop CUDA work. Join it before releasing this worker.
                    task = asyncio.create_task(asyncio.to_thread(self.rife.interpolate, generated))
                    try:
                        return await asyncio.shield(task)
                    except asyncio.CancelledError:
                        await task
                        raise
                await asyncio.sleep(0.1)
            raise TimeoutError("ComfyUI inference exceeded 900 seconds")
        except BaseException:
            await self.cancel()
            raise
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            # Custom output node writes to this service-owned directory, not ComfyUI's permanent output.
            shutil.rmtree(self.data_dir / "outputs" / "liveswap" / window.job_id, ignore_errors=True)

    async def cancel(self):
        try:
            # This ComfyUI instance is exclusively owned by this worker. Remove queued stale work.
            response = await self.http.post(self.url + "/queue", json={"clear": True}, timeout=5)
            response.raise_for_status()
            response = await self.http.post(self.url + "/interrupt", timeout=5)
            response.raise_for_status()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                response = await self.http.get(self.url + "/queue", timeout=5)
                response.raise_for_status()
                if not response.json().get("queue_running"):
                    return
                await asyncio.sleep(0.1)
        except httpx.HTTPError:
            self.unhealthy = True
            return
        self.unhealthy = True

    async def close(self):
        await self.http.aclose()
