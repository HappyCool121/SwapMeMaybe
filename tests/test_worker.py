import asyncio

import httpx
import pytest

from liveswap.archive import pack_window
from liveswap.auth import public_key, sign
from liveswap.transport import WorkerClient
from liveswap.worker.app import Settings, create_app


@pytest.mark.asyncio
async def test_complete_signed_worker_roundtrip(tmp_path, profile, signing_key, inputs):
    settings = Settings(profile, public_key(signing_key), tmp_path / "worker", exit_on_idle=False)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:
            worker = WorkerClient("http://testserver", signing_key, client=http)
            info = await worker.wait_ready(profile, timeout=2)
            assert info["backend"] == "mock"
            window, frames, masks, asset = inputs
            await worker.upload_garment(asset)
            result, output = await worker.process(window, frames, masks, profile)
            assert result.output_ids == list(range(9))
            assert len(output) == 9
            assert not app.state.jobs


@pytest.mark.asyncio
async def test_worker_rejects_unsigned_and_replayed_requests(tmp_path, profile, signing_key):
    app = create_app(Settings(profile, public_key(signing_key), tmp_path / "worker", exit_on_idle=False))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:
            assert (await http.get("/healthz")).status_code == 200
            assert (await http.get("/readyz")).status_code == 401
            headers = sign(signing_key, "GET", "/readyz")
            assert (await http.get("/readyz", headers=headers)).status_code == 200
            assert (await http.get("/readyz", headers=headers)).status_code == 401


@pytest.mark.asyncio
async def test_worker_enforces_single_flight_and_cancellation(tmp_path, profile, signing_key, inputs):
    app = create_app(Settings(profile, public_key(signing_key), tmp_path / "worker", mock_delay=1, exit_on_idle=False))
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:
            worker = WorkerClient("http://testserver", signing_key, client=http)
            window, frames, masks, asset = inputs
            await worker.upload_garment(asset)
            await worker.request("POST", "/jobs", pack_window(window, frames, masks, profile))
            second = window.model_copy(update={"window_id": 1})
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await worker.request("POST", "/jobs", pack_window(second, frames, masks, profile))
            assert caught.value.response.status_code == 409
            await worker.request("DELETE", f"/jobs/{window.job_id}")
            assert app.state.worker_state["active"] is None


@pytest.mark.asyncio
async def test_window_without_uploaded_reference_is_rejected(tmp_path, profile, signing_key, inputs):
    app = create_app(Settings(profile, public_key(signing_key), tmp_path / "worker", exit_on_idle=False))
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:
            worker = WorkerClient("http://testserver", signing_key, client=http)
            window, frames, masks, _ = inputs
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await worker.request("POST", "/jobs", pack_window(window, frames, masks, profile))
            assert caught.value.response.status_code == 404
