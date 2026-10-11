import asyncio
import uuid

import httpx
import pytest

from liveswap.auth import public_key
from liveswap.transport import WorkerClient
from liveswap.worker.app import Settings, create_app


def test_unrelated_data_directory_is_never_cleaned(tmp_path, profile, signing_key):
    assets = tmp_path / "assets"
    assets.mkdir()
    existing = assets / "my-top.png"
    existing.write_bytes(b"user-owned image")
    with pytest.raises(ValueError, match="dedicated"):
        create_app(Settings(profile, public_key(signing_key), tmp_path, exit_on_idle=False))
    assert existing.read_bytes() == b"user-owned image"


@pytest.mark.asyncio
async def test_only_one_session_can_own_worker_at_a_time(tmp_path, profile, signing_key):
    app = create_app(Settings(profile, public_key(signing_key), tmp_path / "worker", exit_on_idle=False))
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as http:
            first, second = WorkerClient("http://testserver", signing_key, client=http), WorkerClient("http://testserver", signing_key, client=http)
            first_session, second_session = uuid.uuid4().hex, uuid.uuid4().hex
            await first.heartbeat(first_session)
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await second.heartbeat(second_session)
            assert caught.value.response.status_code == 409
            await first.close()
            await second.heartbeat(second_session)
            await second.close()
