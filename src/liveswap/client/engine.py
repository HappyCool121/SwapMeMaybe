from __future__ import annotations

import asyncio
import fcntl
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from liveswap.archive import prepare_garment
from liveswap.auth import load_key
from liveswap.client.capture import AudioBuffer, CameraSource, FileSource, Microphone, SyntheticSource
from liveswap.client.output import FFmpegOutput
from liveswap.client.vision import MediaPipeMasker, SyntheticMasker
from liveswap.contracts import Profile, Window, capacity, frame_pts
from liveswap.timeline import (
    Fallback,
    FrameBuffer,
    SourceFrame,
    Stitcher,
    WindowAssembler,
    crop_for,
    window_images,
)
from liveswap.transport import WorkerClient


@dataclass
class ClientOptions:
    profile: Profile
    worker_urls: list[str] = field(default_factory=list)
    key_path: Path = Path(".liveswap/client.pem")
    garment_path: Path | None = None
    model_dir: Path = Path("models/mediapipe")
    camera: int = 0
    microphone: int | None = None
    microphone_delay_ms: float = 0
    no_microphone: bool = False
    video: Path | None = None
    synthetic: bool = False
    passthrough: bool = False
    delay: float = 8.0
    duration: float = 0
    output: str = "tcp://127.0.0.1:10000?listen=1"
    codec: str = "auto"
    benchmark: Path | None = None
    experimental: bool = False
    metrics_path: Path | None = Path("artifacts/session.json")


class Session:
    def __init__(self, options: ClientOptions, *, stop: threading.Event | None = None,
                 status: Callable[[dict], None] | None = None, preview: Callable[[np.ndarray], None] | None = None):
        self.options = options
        self.stop = stop or threading.Event()
        self.status_callback, self.preview_callback = status, preview
        self.session_id = uuid.uuid4().hex
        self.metrics = {"session_id": self.session_id, "captured_frames": 0, "played_frames": 0,
                        "edited_frames": 0, "missing_frames": 0, "tracking_misses": 0,
                        "submitted_windows": 0, "completed_windows": 0, "dropped_windows": 0,
                        "failed_windows": 0, "last_error": None, "peak_source_frames": 0}
        self.workers: list[WorkerClient] = []
        self.predicted = 0.0
        self.report = None
        self.garment_data: bytes | None = None

    def _preflight(self, worker_count: int | None = None):
        o = self.options
        count = len(o.worker_urls) if worker_count is None else worker_count
        if not 5 <= o.delay <= 10:
            raise ValueError("fixed delay must be between five and ten seconds")
        if o.duration < 0:
            raise ValueError("duration must be positive, or zero for an unlimited stream")
        if not -500 <= o.microphone_delay_ms <= 500:
            raise ValueError("microphone sync adjustment must be between -500 and 500 milliseconds")
        if not o.passthrough and (count < 1 or o.garment_path is None):
            raise ValueError("select a garment and at least one worker")
        if o.synthetic and o.profile.backend != "mock":
            raise ValueError("synthetic mode requires the distinctly labeled mock profile")
        if o.profile.backend == "mock" and not o.synthetic:
            raise ValueError("mock workers are restricted to synthetic development scenes")
        if o.passthrough or o.synthetic:
            return
        if o.benchmark:
            report = json.loads(o.benchmark.read_text())
            self.report = report
            if report["profile_id"] != o.profile.id or report["backend"] != "comfy":
                raise ValueError("benchmark profile/backend mismatch")
            self.predicted = report["turnaround_p95_seconds"]
            limits = capacity(o.profile, self.predicted, count)
            if "worker_hourly_price_usd" in report:
                self.metrics["estimated_pool_hourly_price_usd"] = report["worker_hourly_price_usd"] * count
            if not o.experimental:
                if report["samples"] < 10 or not report.get("visual_review_pass", False):
                    raise ValueError("production requires ten benchmark samples and a recorded visual review; use --experimental for initial qualification")
                if not limits["delay_pass"] or not limits["throughput_pass"] or o.delay < limits["required_fixed_delay_seconds"]:
                    raise ValueError(f"benchmark fails the selected delay or pool capacity: {limits}")
        elif not o.experimental:
            raise ValueError("GPU streams require a benchmark report or explicit --experimental mode")

    def emit(self, **updates):
        self.metrics.update(updates)
        if self.status_callback:
            self.status_callback(dict(self.metrics))

    async def run(self) -> dict:
        self._preflight()
        o, profile = self.options, self.options.profile
        source = masker = microphone = output = None
        tasks = []
        self.emit(phase="Starting workers", fixed_delay_seconds=o.delay,
                  backend="passthrough" if o.passthrough else profile.backend, profile_id=profile.id)
        try:
            if not o.passthrough:
                key = load_key(o.key_path)
                garment_data = self.garment_data or prepare_garment(o.garment_path)
                self.workers = [WorkerClient(url, key) for url in o.worker_urls]
                async def keep_leases():
                    while True:
                        await asyncio.gather(*(worker.heartbeat(self.session_id) for worker in self.workers), return_exceptions=True)
                        await asyncio.sleep(10)
                tasks.append(asyncio.create_task(keep_leases()))
                information = await asyncio.gather(*(worker.wait_ready(profile) for worker in self.workers))
                await asyncio.gather(*(worker.heartbeat(self.session_id) for worker in self.workers))
                provenances = {json.dumps(info.get("provenance", {}), sort_keys=True) for info in information}
                if len(provenances) != 1:
                    raise ValueError("workers have different code/model provenance; use one immutable image")
                if self.report and not o.experimental and self.report.get("worker_provenance", {}) != information[0].get("provenance", {}):
                    raise ValueError("benchmark was produced by a different worker image/model bundle")
                hashes = await asyncio.gather(*(worker.upload_garment(garment_data) for worker in self.workers))
                garment_hash = hashes[0]
            else:
                garment_hash = "0" * 64
            source = SyntheticSource() if o.synthetic else FileSource(o.video) if o.video else CameraSource(o.camera)
            masker = SyntheticMasker() if o.synthetic else MediaPipeMasker(o.model_dir)
            if isinstance(source, CameraSource):
                until = time.monotonic() + 5
                while source.read(0) is None:
                    if time.monotonic() > until:
                        raise RuntimeError("camera did not deliver frames")
                    await asyncio.sleep(0.05)
            output = FFmpegOutput(o.output, o.codec)
            if output.is_live:
                self.emit(phase="Waiting for OBS to connect", obs_source_url=output.receiver_url)
                priming = asyncio.create_task(asyncio.to_thread(output.prime, self.stop))
                try:
                    await asyncio.shield(priming)
                except asyncio.CancelledError:
                    self.stop.set()
                    await asyncio.gather(priming, return_exceptions=True)
                    raise
            epoch = time.monotonic()
            audio = AudioBuffer()
            if not o.no_microphone and not o.synthetic and o.video is None:
                microphone = Microphone(audio, epoch, o.microphone)
            buffer = FrameBuffer()
            stitcher = Stitcher(profile, self.session_id)
            fallback = Fallback()
            assembler = WindowAssembler(profile)
            capture_queue: asyncio.Queue[SourceFrame | None] = asyncio.Queue(maxsize=4)
            windows = asyncio.PriorityQueue(maxsize=max(2, 2 * len(self.workers)))
            capture_done = asyncio.Event()
            next_window = 0

            async def enqueue(frames):
                nonlocal next_window
                if o.passthrough:
                    return
                deadline = epoch + frames[0].frame_id / 24 + o.delay - 0.05
                if deadline <= time.monotonic():
                    self.metrics["dropped_windows"] += 1
                    return
                crop = await asyncio.to_thread(crop_for, frames)
                images, masks = await asyncio.to_thread(window_images, frames, crop, profile)
                window = Window(session_id=self.session_id, window_id=next_window,
                                profile_id=profile.id, garment_hash=garment_hash,
                                frame_ids=[frame.frame_id for frame in frames],
                                pts_us=[frame_pts(frame.frame_id) for frame in frames], crop=crop,
                                budget_ms=max(1, min(120_000, round((deadline - time.monotonic()) * 1000))))
                next_window += 1
                try:
                    windows.put_nowait((deadline, window.window_id, window, images, masks))
                except asyncio.QueueFull:
                    self.metrics["dropped_windows"] += 1

            async def capture():
                frame_id = 0
                try:
                    while not self.stop.is_set() and (not o.duration or frame_id < round(o.duration * 24)):
                        due = epoch + frame_id / 24
                        await asyncio.sleep(max(0, due - time.monotonic()))
                        reading = asyncio.create_task(asyncio.to_thread(source.read, frame_id))
                        try:
                            rgb = await asyncio.shield(reading)
                        except asyncio.CancelledError:
                            await reading
                            raise
                        if rgb is None:
                            if isinstance(source, FileSource):
                                break
                            # Camera stalls remain missing timestamps, never a shifted playback clock.
                            rgb = fallback.slate.copy()
                            valid = False
                        else:
                            valid = True
                        frame = SourceFrame(frame_id, rgb, None)
                        buffer.put(frame)
                        self.metrics["captured_frames"] = frame_id + 1
                        self.metrics["peak_source_frames"] = max(self.metrics["peak_source_frames"], len(buffer.frames))
                        if valid:
                            if capture_queue.full():
                                capture_queue.get_nowait()
                                self.metrics["tracking_misses"] += 1
                            capture_queue.put_nowait(frame)
                        frame_id += 1
                finally:
                    if capture_queue.full():
                        capture_queue.get_nowait()
                    capture_queue.put_nowait(None)
                    capture_done.set()

            async def preprocess():
                while True:
                    frame = await capture_queue.get()
                    if frame is None:
                        tail = assembler.finish()
                        if tail:
                            await enqueue(tail)
                        return
                    analysis = asyncio.create_task(asyncio.to_thread(masker.mask, frame.rgb, frame_pts(frame.frame_id)))
                    try:
                        frame.mask = await asyncio.shield(analysis)
                    except asyncio.CancelledError:
                        await analysis
                        raise
                    if frame.mask is None:
                        self.metrics["tracking_misses"] += 1
                    prepared = assembler.push(frame)
                    if prepared:
                        await enqueue(prepared)

            async def infer(worker):
                while True:
                    deadline, _, window, images, masks = await windows.get()
                    remaining = deadline - time.monotonic()
                    predicted = float(np.percentile(worker.turnarounds, 95)) if worker.turnarounds else self.predicted
                    if remaining <= max(0.1, predicted):
                        self.metrics["dropped_windows"] += 1
                        continue
                    window = window.model_copy(update={"budget_ms": max(1, round(remaining * 1000))})
                    self.metrics["submitted_windows"] += 1
                    try:
                        result, frames = await worker.process(window, images, masks, profile)
                        stitcher.add(result, frames)
                        self.metrics["completed_windows"] += 1
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.metrics["failed_windows"] += 1
                        self.metrics["last_error"] = str(exc)
                        # Re-establish reference/readiness after a replaced worker, without retrying late windows.
                        try:
                            await worker.wait_ready(profile, timeout=5)
                            await worker.upload_garment(garment_data)
                        except Exception:
                            await asyncio.sleep(1)

            async def playback():
                frame_id = 0
                last_preview = -10.0
                while not self.stop.is_set():
                    due = epoch + o.delay + frame_id / 24
                    await asyncio.sleep(max(0, due - time.monotonic()))
                    if capture_done.is_set() and frame_id >= self.metrics["captured_frames"]:
                        return
                    if time.monotonic() - due > 0.25:
                        raise RuntimeError("client cannot maintain the fixed playback clock; reduce local load")
                    self.metrics["max_playback_lateness_seconds"] = max(self.metrics.get("max_playback_lateness_seconds", 0), time.monotonic() - due)
                    original = buffer.pop(frame_id)
                    edited = original.rgb.copy() if o.passthrough and original else stitcher.composite(original) if original else None
                    rgb, available = fallback.render(edited)
                    if o.synthetic or o.passthrough:
                        image = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 670, 1280, 720), fill=(120, 35, 35))
                        draw.text((20, 682), "TRANSPORT TEST — NO CLOTHING REPLACEMENT", fill="white", font_size=24)
                        rgb = np.asarray(image)
                    samples = audio.read(frame_id * 2000 - round(o.microphone_delay_ms * 48), 2000)
                    await asyncio.to_thread(output.send, rgb, samples)
                    self.metrics["played_frames"] += 1
                    self.metrics["edited_frames" if available else "missing_frames"] += 1
                    if frame_id % 24 == 0:
                        useful = sum(profile.stride / np.percentile(worker.turnarounds, 95) for worker in self.workers if worker.turnarounds)
                        self.emit(phase="Streaming", measured_useful_fps=round(float(useful), 2),
                                  audio_status_events=microphone.status_count if microphone else 0)
                    if self.preview_callback and frame_id / 24 - last_preview >= 0.25:
                        self.preview_callback(rgb.copy())
                        last_preview = frame_id / 24
                    frame_id += 1

            async def fill_live_output():
                for index in range(round(o.delay * 24)):
                    await asyncio.sleep(max(0, epoch + index / 24 - time.monotonic()))
                    await asyncio.to_thread(output.send, fallback.slate, np.zeros(2000, np.float32))

            tasks.extend([asyncio.create_task(capture()), asyncio.create_task(preprocess())])
            capture_task, preprocessing_task = tasks[-2:]
            filling = asyncio.create_task(fill_live_output()) if output.is_live else None
            if filling:
                tasks.append(filling)
            tasks.extend(asyncio.create_task(infer(worker)) for worker in self.workers)
            player = asyncio.create_task(playback())
            tasks.append(player)
            self.emit(phase="Filling fixed-delay buffer")
            # Surface capture/preprocessing failures immediately; inference failures use the specified fallback.
            watched = [player, capture_task, preprocessing_task]
            if filling:
                watched.append(filling)
            while not player.done():
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception():
                        raise task.exception()
                    watched.remove(task)
            await player
            self.emit(phase="Stopped")
            return self.metrics
        except BaseException as exc:
            self.emit(phase="Stopped" if isinstance(exc, asyncio.CancelledError) else "Error", last_error=str(exc))
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            cleanup_errors = []
            for component in (microphone, source, masker, output):
                if component is not None:
                    try:
                        await asyncio.to_thread(component.close)
                    except Exception as exc:
                        cleanup_errors.append(f"{type(component).__name__}: {exc}")
            results = await asyncio.gather(*(worker.close() for worker in self.workers), return_exceptions=True)
            cleanup_errors.extend(str(result) for result in results if isinstance(result, BaseException))
            if cleanup_errors:
                self.metrics["cleanup_errors"] = cleanup_errors
            if o.metrics_path:
                o.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                o.metrics_path.write_text(json.dumps(self.metrics, indent=2) + "\n")


async def run_managed(options: ClientOptions, *, deployment: Path | None = None, stop=None, status=None, preview=None):
    """A managed session owns its pool lifecycle, including failure recovery and teardown."""
    from liveswap.auth import public_key
    from liveswap.nosana import NosanaPool

    pool = None
    maintenance = None
    ownership = None
    owns_pool = False
    session = Session(options, stop=stop, status=status, preview=preview)
    parent = asyncio.current_task()

    async def watch_stop():
        while not session.stop.is_set():
            await asyncio.sleep(0.1)
        parent.cancel()

    stopping = asyncio.create_task(watch_stop())
    try:
        if deployment:
            if options.worker_urls:
                raise ValueError("use --deployment or --worker, not both")
            ownership = deployment.with_suffix(".session.lock").open("a")
            try:
                fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("this managed pool already has an active local session") from exc
            pool = NosanaPool(deployment)
            if pool.data["profile_id"] != options.profile.id or pool.data["public_key"] != public_key(load_key(options.key_path)):
                raise ValueError("deployment profile or signing key does not match")
            session._preflight(worker_count=len(pool.data["deployments"]))
            session.garment_data = prepare_garment(options.garment_path)
            owns_pool = True
            options.worker_urls = await pool.start_and_resolve()

            async def maintain():
                while not session.workers:
                    await asyncio.sleep(0.2)
                await pool.maintain(session.workers)
            maintenance = asyncio.create_task(maintain())
        return await session.run()
    except asyncio.CancelledError:
        if session.stop.is_set():
            return session.metrics
        raise
    finally:
        stopping.cancel()
        await asyncio.gather(stopping, return_exceptions=True)
        if maintenance:
            maintenance.cancel()
            await asyncio.gather(maintenance, return_exceptions=True)
        try:
            if pool:
                try:
                    if owns_pool:
                        await pool.stop()
                finally:
                    await pool.close()
        finally:
            if ownership:
                fcntl.flock(ownership, fcntl.LOCK_UN)
                ownership.close()
