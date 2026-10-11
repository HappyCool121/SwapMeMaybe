from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import re
import signal
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

from liveswap.archive import MAX_ARCHIVE, decode_image, pack_result, unpack_window
from liveswap.auth import Verifier
from liveswap.contracts import Profile, Result, Window
from liveswap.worker.backends import ComfyBackend, MockBackend


@dataclass
class Settings:
    profile: Profile
    public_key: str
    data_dir: Path
    comfy_url: str = "http://127.0.0.1:8188"
    rife_repo: Path = Path("/opt/rife")
    rife_weights: Path = Path("/models/rife")
    mock_delay: float = 0
    result_ttl: float = 60
    lease_seconds: float = 120
    exit_on_idle: bool = True

    @classmethod
    def from_env(cls):
        profile = Profile.model_validate_json(os.environ["LIVESWAP_PROFILE_JSON"]) if "LIVESWAP_PROFILE_JSON" in os.environ else Profile.load(Path(os.environ.get("LIVESWAP_PROFILE", "profiles/vace.json")))
        return cls(profile=profile, public_key=os.environ["LIVESWAP_PUBLIC_KEY"],
                   data_dir=Path(os.environ.get("LIVESWAP_DATA_DIR", "/data/liveswap")),
                   comfy_url=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"),
                   rife_repo=Path(os.environ.get("RIFE_REPO", "/opt/rife")),
                   rife_weights=Path(os.environ.get("RIFE_WEIGHTS", "/models/rife")),
                   mock_delay=float(os.environ.get("LIVESWAP_MOCK_DELAY", "0")),
                   exit_on_idle=os.environ.get("LIVESWAP_EXIT_ON_IDLE", "1") == "1")


@dataclass
class Job:
    window: Window
    created: float
    state: str = "running"
    result: bytes | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    finished: float | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    profile = settings.profile
    verifier = Verifier(settings.public_key)
    root = settings.data_dir
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".liveswap-worker-owned"
    if not marker.exists() and any(root.iterdir()):
        raise ValueError("worker data_dir must be empty or previously owned by LiveSwap; use a dedicated directory")
    marker.write_text("LiveSwap temporary worker data\n")
    ownership = (root / "worker.lock").open("a")
    try:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        ownership.close()
        raise RuntimeError("another worker owns this data_dir; use a separate directory") from exc
    for folder in ("assets", "jobs", "outputs"):
        import shutil
        shutil.rmtree(root / folder, ignore_errors=True)
        (root / folder).mkdir(exist_ok=True)
    backend = MockBackend(profile, settings.mock_delay) if profile.backend == "mock" else ComfyBackend(profile, root, settings.comfy_url, settings.rife_repo, settings.rife_weights)
    jobs: dict[str, Job] = {}
    lock = asyncio.Lock()
    state = {"ready": False, "error": None, "active": None, "last_contact": time.monotonic(), "claimed": False,
             "consecutive_failures": 0, "session_id": None, "garment_hash": None}
    worker_id = uuid.uuid4().hex

    async def warm():
        try:
            await backend.warm()
            state["ready"] = True
            state["last_contact"] = time.monotonic()
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"

    async def housekeeping():
        while True:
            await asyncio.sleep(1)
            now = time.monotonic()
            expired = [key for key, job in jobs.items() if job.finished is not None and now - job.finished > settings.result_ttl]
            for key in expired:
                del jobs[key]
            limit = settings.lease_seconds if state["claimed"] else 1200
            if settings.exit_on_idle and now - state["last_contact"] > limit:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    @asynccontextmanager
    async def lifespan(app):
        warming = asyncio.create_task(warm())
        cleaning = asyncio.create_task(housekeeping())
        try:
            yield
        finally:
            pending = [warming, cleaning] + [job.task for job in jobs.values() if job.task and not job.task.done()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await backend.close()
            for folder in ("assets", "jobs", "outputs"):
                import shutil
                shutil.rmtree(root / folder, ignore_errors=True)
            fcntl.flock(ownership, fcntl.LOCK_UN)
            ownership.close()

    app = FastAPI(title="LiveSwap worker", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.worker_state, app.state.jobs = state, jobs

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/healthz" and request.method == "GET":
            return await call_next(request)
        try:
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_ARCHIVE:
                    return JSONResponse({"detail": "upload exceeds limit"}, status_code=413)
                chunks.append(chunk)
            request.state.body = b"".join(chunks)
            verifier.verify(request.method, request.url.path, request.state.body, request.headers)
        except ValueError:
            return JSONResponse({"detail": "invalid or replayed signature"}, status_code=401)
        state["last_contact"], state["claimed"] = time.monotonic(), True
        return await call_next(request)

    @app.get("/healthz")
    async def health():
        return {"alive": True}

    @app.get("/readyz")
    async def ready():
        provenance = {"backend": "mock"} if profile.backend == "mock" else {
            "comfy_commit": os.environ.get("COMFY_COMMIT", "unmanaged"),
            "rife_commit": os.environ.get("RIFE_COMMIT", "unmanaged"),
            "wan_revision": os.environ.get("WAN_REVISION", "unmanaged"),
            "app_source_sha256": os.environ.get("LIVESWAP_SOURCE_SHA256", "unmanaged"),
            "rife_weights_sha256": getattr(backend.rife, "weights_sha256", None),
        }
        return {"ready": state["ready"], "error": state["error"], "worker_id": worker_id,
                "profile_id": profile.id, "profile": profile.model_dump(), "backend": profile.backend,
                "busy": state["active"] is not None, "provenance": provenance, "active_session": state["session_id"]}

    @app.put("/assets/{digest}")
    async def asset(digest: str, request: Request):
        body = request.state.body
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or hashlib.sha256(body).hexdigest() != digest:
            raise HTTPException(400, "garment hash mismatch")
        if len(body) > 10 * 1024 * 1024:
            raise HTTPException(413, "garment image exceeds 10 MiB")
        path = root / "assets" / f"{digest}.png"
        if not path.exists() and len(list((root / "assets").glob("*.png"))) >= 8:
            raise HTTPException(409, "asset cache full; restart the worker between sessions")
        try:
            image = decode_image(body)
        except Exception as exc:
            raise HTTPException(400, "invalid garment image") from exc
        Image.fromarray(image).save(path)
        return {"hash": digest}

    async def execute(job: Job, frames: list[np.ndarray], masks: list[np.ndarray]):
        try:
            async with asyncio.timeout(job.window.budget_ms / 1000):
                frames = await backend.generate(job.window, frames, masks, root / "assets" / f"{job.window.garment_hash}.png")
                result = Result(window=job.window, backend=profile.backend, output_ids=job.window.output_ids,
                                processing_seconds=time.monotonic() - job.created)
                job.result = pack_result(result, frames, profile.jpeg_quality)
                job.state = "complete"
                state["consecutive_failures"] = 0
        except asyncio.CancelledError:
            job.state, job.error = "failed", "cancelled"
            raise
        except TimeoutError:
            job.state, job.error = "failed", "processing budget exceeded"
        except Exception as exc:
            job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
            state["consecutive_failures"] += 1
            if state["consecutive_failures"] >= 3:
                state["ready"], state["error"] = False, "repeated inference failures; restart this worker"
        finally:
            job.finished = time.monotonic()
            state["active"] = None
            if getattr(backend, "unhealthy", False):
                state["ready"], state["error"] = False, "ComfyUI did not stop cleanly; restart this worker"

    @app.post("/jobs", status_code=202)
    async def submit(request: Request):
        if not state["ready"]:
            raise HTTPException(503, "worker is warming up")
        async with lock:
            if state["active"] is not None or len(jobs) >= 32:
                raise HTTPException(409, "worker busy or result capacity reached")
            try:
                window, frames, masks = unpack_window(request.state.body, profile)
            except Exception as exc:
                raise HTTPException(400, "invalid window or incompatible profile") from exc
            if not (root / "assets" / f"{window.garment_hash}.png").exists():
                raise HTTPException(404, "upload garment before submitting windows")
            if state["session_id"] not in (None, window.session_id):
                raise HTTPException(409, "worker belongs to another active session")
            if state["garment_hash"] not in (None, window.garment_hash):
                raise HTTPException(409, "restart the session to change garments")
            if window.job_id in jobs:
                raise HTTPException(409, "window was already submitted")
            state["session_id"], state["garment_hash"] = window.session_id, window.garment_hash
            job = Job(window=window, created=time.monotonic())
            jobs[window.job_id] = job
            state["active"] = window.job_id
            job.task = asyncio.create_task(execute(job, frames, masks))
            return {"job_id": window.job_id}

    def lookup(job_id):
        if job_id not in jobs:
            raise HTTPException(404, "unknown or expired job")
        return jobs[job_id]

    @app.get("/jobs/{job_id}")
    async def status(job_id: str):
        job = lookup(job_id)
        return {"state": job.state, "error": job.error}

    @app.get("/jobs/{job_id}/result")
    async def result(job_id: str):
        job = lookup(job_id)
        if job.state != "complete":
            raise HTTPException(409, "result is not complete")
        return Response(job.result, media_type="application/zip")

    @app.delete("/jobs/{job_id}")
    async def release(job_id: str):
        job = lookup(job_id)
        if job.task and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        if state["active"] == job_id:
            state["active"] = None
        jobs.pop(job_id, None)
        return {"released": True}

    @app.post("/sessions/{session_id}/heartbeat")
    async def heartbeat(session_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise HTTPException(400, "invalid session identifier")
        if state["session_id"] not in (None, session_id):
            raise HTTPException(409, "worker belongs to another active session")
        state["session_id"] = session_id
        return {"lease_seconds": settings.lease_seconds}

    @app.delete("/sessions/{session_id}")
    async def release_session(session_id: str):
        if state["session_id"] != session_id:
            return {"released": False}
        for job_id, job in list(jobs.items()):
            if job.window.session_id == session_id:
                if job.task and not job.task.done():
                    job.task.cancel()
                    await asyncio.gather(job.task, return_exceptions=True)
                jobs.pop(job_id, None)
        state["active"], state["session_id"], state["garment_hash"] = None, None, None
        return {"released": True}

    return app
