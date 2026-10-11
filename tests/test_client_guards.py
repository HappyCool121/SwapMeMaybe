import asyncio
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from liveswap.client.engine import ClientOptions, Session, run_managed
from liveswap.client.vision import hand_protection
from liveswap.contracts import Profile


def test_mock_cannot_be_used_on_a_real_camera(profile):
    options = ClientOptions(profile=profile, worker_urls=["http://localhost:8000"], garment_path=Path("shirt.jpg"))
    with pytest.raises(ValueError, match="synthetic"):
        Session(options)._preflight()


def test_unmeasured_gpu_stream_requires_experimental_mode():
    options = ClientOptions(profile=Profile(), worker_urls=["https://worker.example"], garment_path=Path("shirt.jpg"))
    with pytest.raises(ValueError, match="benchmark"):
        Session(options)._preflight()


def test_production_rejects_slow_configuration(tmp_path):
    profile = Profile()
    report = tmp_path / "benchmark.json"
    report.write_text(json.dumps({"profile_id": profile.id, "backend": "comfy", "samples": 10,
                                  "visual_review_pass": True, "turnaround_p95_seconds": 15}))
    options = ClientOptions(profile=profile, worker_urls=["https://worker.example"] * 32,
                            garment_path=Path("shirt.jpg"), benchmark=report)
    with pytest.raises(ValueError, match="capacity"):
        Session(options)._preflight()


def test_pose_hands_protect_region_without_erasing_entire_shirt():
    pytest.importorskip("cv2")
    points = np.zeros((33, 2))
    points[[15, 17, 19, 21]] = [[0.4, 0.5], [0.45, 0.55], [0.5, 0.5], [0.45, 0.45]]
    visibility = [0.0] * 33
    for index in [15, 17, 19, 21]:
        visibility[index] = 1
    mask = hand_protection(points, visibility, 100, 100)
    assert mask[50, 45]
    assert not mask[80, 80]
    assert not mask[0, 0]


def test_portrait_source_is_letterboxed_without_stretching():
    pytest.importorskip("cv2")
    from liveswap.client.capture import normalize_frame
    bgr = np.full((800, 400, 3), (10, 20, 30), np.uint8)
    rgb = normalize_frame(bgr)
    assert rgb.shape == (720, 1280, 3)
    assert np.array_equal(rgb[360, 640], [30, 20, 10])
    assert np.array_equal(rgb[360, 0], [24, 24, 24])
    assert np.all(rgb[:, 460:820] == [30, 20, 10])


@pytest.mark.asyncio
async def test_stop_works_while_waiting_for_worker(monkeypatch, profile, tmp_path):
    from liveswap.auth import create_key
    from liveswap.transport import WorkerClient
    key = tmp_path / "client.pem"
    create_key(key)
    garment = tmp_path / "shirt.jpg"
    from PIL import Image
    Image.new("RGB", (16, 16), "blue").save(garment)

    async def stall(*args, **kwargs):
        await asyncio.sleep(100)

    monkeypatch.setattr(WorkerClient, "wait_ready", stall)
    stop = threading.Event()
    options = ClientOptions(profile=profile, worker_urls=["http://localhost:8000"], garment_path=garment,
                            key_path=key, synthetic=True, metrics_path=None)
    task = asyncio.create_task(run_managed(options, stop=stop))
    await asyncio.sleep(0.03)
    stop.set()
    result = await asyncio.wait_for(task, timeout=2)
    assert result["phase"] == "Stopped"


@pytest.mark.asyncio
async def test_invalid_preflight_never_starts_or_stops_a_paid_pool(monkeypatch, tmp_path):
    from liveswap.auth import create_key, load_key, public_key
    key = tmp_path / "client.pem"
    create_key(key)
    profile = Profile()
    calls = []

    class Pool:
        def __init__(self, path):
            self.data = {"profile_id": profile.id, "public_key": public_key(load_key(key)), "deployments": [{"id": "owned"}]}

        async def start_and_resolve(self):
            calls.append("start")
            return ["https://worker.example"]

        async def stop(self):
            calls.append("stop")

        async def close(self):
            calls.append("close")

    monkeypatch.setattr("liveswap.nosana.NosanaPool", Pool)
    options = ClientOptions(profile=profile, key_path=key, garment_path=Path("shirt.png"))
    with pytest.raises(ValueError, match="benchmark"):
        await run_managed(options, deployment=tmp_path / "pool.json")
    assert calls == ["close"]
