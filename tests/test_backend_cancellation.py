import httpx
import pytest

from liveswap.worker.backends import ComfyBackend


@pytest.mark.asyncio
async def test_cancellation_clears_stale_comfy_queue_before_releasing_worker(profile, tmp_path):
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"queue_running": [], "queue_pending": []})

    backend = ComfyBackend(profile, tmp_path, "http://localhost:8188", tmp_path, tmp_path)
    await backend.http.aclose()
    backend.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await backend.cancel()
    finally:
        await backend.close()
    assert calls == [("POST", "/queue"), ("POST", "/interrupt"), ("GET", "/queue")]
    assert not backend.unhealthy


@pytest.mark.asyncio
async def test_unreachable_comfy_is_fenced_off(profile, tmp_path):
    def handle(request):
        raise httpx.ConnectError("worker backend disappeared")

    backend = ComfyBackend(profile, tmp_path, "http://localhost:8188", tmp_path, tmp_path)
    await backend.http.aclose()
    backend.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await backend.cancel()
    finally:
        await backend.close()
    assert backend.unhealthy
