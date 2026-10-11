from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import numpy as np

from liveswap.archive import prepare_garment
from liveswap.auth import load_key
from liveswap.client.capture import FileSource, SyntheticSource
from liveswap.client.vision import MediaPipeMasker, SyntheticMasker
from liveswap.contracts import Profile, Window, capacity, frame_pts
from liveswap.timeline import SourceFrame, crop_for, window_images
from liveswap.transport import WorkerClient


async def benchmark(*, url: str, key: Path, profile: Profile, garment: Path, samples: int,
                    report_path: Path, video: Path | None = None, synthetic: bool = False,
                    model_dir: Path = Path("models/mediapipe"), hourly_price: float | None = None) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    if synthetic != (profile.backend == "mock"):
        raise ValueError("synthetic benchmarks require mock workers; genuine GPU benchmarks require a video")
    if not synthetic and video is None:
        raise ValueError("provide representative footage with --video for GPU qualification")
    worker = WorkerClient(url, load_key(key))
    source = SyntheticSource() if synthetic else FileSource(video)
    masker = SyntheticMasker() if synthetic else MediaPipeMasker(model_dir)
    session_id = uuid.uuid4().hex
    durations, frame_id = [], 0
    try:
        info = await worker.wait_ready(profile)
        await worker.heartbeat(session_id)
        digest = await worker.upload_garment(prepare_garment(garment))
        for sample in range(samples):
            frames = []
            while len(frames) < profile.length:
                rgb = await asyncio.to_thread(source.read, frame_id)
                if rgb is None:
                    raise RuntimeError("benchmark video ended before enough valid windows were collected")
                mask = await asyncio.to_thread(masker.mask, rgb, frame_pts(frame_id))
                if mask is None:
                    frames.clear()
                else:
                    frames.append(SourceFrame(frame_id, rgb, mask))
                frame_id += 2
            crop = crop_for(frames)
            images, masks = window_images(frames, crop, profile)
            window = Window(session_id=session_id, window_id=sample, profile_id=profile.id, garment_hash=digest,
                            frame_ids=[frame.frame_id for frame in frames], pts_us=[frame_pts(frame.frame_id) for frame in frames],
                            crop=crop, budget_ms=120_000)
            started = time.monotonic()
            await worker.process(window, images, masks, profile)
            durations.append(time.monotonic() - started)
            print(f"sample {sample + 1}/{samples}: {durations[-1]:.3f}s", flush=True)
        p95 = float(np.percentile(durations, 95))
        result = {"profile_id": profile.id, "profile": profile.model_dump(), "backend": profile.backend,
                  "samples": len(durations), "turnarounds_seconds": durations,
                  "turnaround_p50_seconds": float(np.median(durations)), "turnaround_p95_seconds": p95,
                  "worker_provenance": info.get("provenance", {}), "visual_review_pass": False,
                  "synthetic": synthetic, "capacity": capacity(profile, p95)}
        if hourly_price is not None:
            result["worker_hourly_price_usd"] = hourly_price
            result["estimated_pool_hourly_price_usd"] = hourly_price * result["capacity"]["workers_with_25_percent_headroom"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        source.close()
        masker.close()
        await worker.close()
