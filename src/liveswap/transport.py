from __future__ import annotations

import asyncio
import hashlib
import time
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from liveswap.archive import pack_window, unpack_result
from liveswap.auth import sign
from liveswap.contracts import Profile, Window


class WorkerClient:
    def __init__(self, url: str, key: Ed25519PrivateKey, *, client: httpx.AsyncClient | None = None):
        parsed = urlparse(url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1", "testserver"}):
            raise ValueError("remote workers require HTTPS; plain HTTP is allowed only on localhost")
        self.url = url.rstrip("/")
        self.key = key
        self.http = client or httpx.AsyncClient(timeout=30, follow_redirects=False)
        self.owns_client = client is None
        self.busy = False
        self.turnarounds: list[float] = []
        self.last_error: str | None = None
        self.session_id: str | None = None

    async def request(self, method: str, path: str, body: bytes = b"", **kwargs) -> httpx.Response:
        response = await self.http.request(method, self.url + path, content=body, headers=sign(self.key, method, path, body), **kwargs)
        response.raise_for_status()
        return response

    async def ready(self) -> dict:
        return (await self.request("GET", "/readyz")).json()

    async def wait_ready(self, profile: Profile, timeout: float = 900) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                info = await self.ready()
                if info["profile_id"] != profile.id or info["backend"] != profile.backend:
                    raise ValueError("worker profile does not match the client's profile")
                if info["ready"]:
                    return info
                if info.get("error"):
                    raise RuntimeError(f"worker warmup failed: {info['error']}")
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403}:
                    raise RuntimeError("worker rejected the signing key") from exc
            await asyncio.sleep(1)
        raise TimeoutError("worker did not become ready before startup timeout")

    async def upload_garment(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        await self.request("PUT", f"/assets/{digest}", data)
        return digest

    async def heartbeat(self, session_id: str) -> None:
        await self.request("POST", f"/sessions/{session_id}/heartbeat")
        self.session_id = session_id

    async def process(self, window: Window, frames, masks, profile: Profile):
        self.session_id = window.session_id
        started = time.monotonic()
        deadline = started + window.budget_ms / 1000
        job_id = window.job_id
        accepted = False
        try:
            async with asyncio.timeout(window.budget_ms / 1000):
                bundle = await asyncio.to_thread(pack_window, window, frames, masks, profile)
                await self.request("POST", "/jobs", bundle)
                accepted = True
                while time.monotonic() < deadline:
                    status = (await self.request("GET", f"/jobs/{job_id}")).json()
                    if status["state"] == "complete":
                        response = await self.request("GET", f"/jobs/{job_id}/result")
                        result, output = await asyncio.to_thread(unpack_result, response.content, profile)
                        if result.window != window:
                            raise ValueError("worker returned a different window manifest")
                        self.turnarounds.append(time.monotonic() - started)
                        self.turnarounds = self.turnarounds[-100:]
                        self.last_error = None
                        return result, output
                    if status["state"] == "failed":
                        raise RuntimeError(status.get("error", "worker inference failed"))
                    await asyncio.sleep(0.1)
                raise TimeoutError("window missed its deadline")
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            if accepted:
                try:
                    await self.request("DELETE", f"/jobs/{job_id}", timeout=3)
                except (httpx.HTTPError, asyncio.CancelledError):
                    pass

    async def close(self):
        if self.session_id and not self.http.is_closed:
            try:
                await self.request("DELETE", f"/sessions/{self.session_id}", timeout=3)
            except httpx.HTTPError:
                pass
        if self.owns_client:
            await self.http.aclose()
