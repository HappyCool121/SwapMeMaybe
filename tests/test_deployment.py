import json

import httpx
import pytest

from liveswap.auth import public_key
from liveswap.nosana import NosanaAPI, job_definition, provision
from liveswap.worker.graph import build_graph


def test_public_job_contains_only_public_key(profile, signing_key):
    body = job_definition("example/worker@sha256:" + "a" * 64, profile, public_key(signing_key))
    encoded = json.dumps(body)
    assert "PRIVATE KEY" not in encoded
    assert "NOSANA_API_KEY" not in encoded
    assert body["ops"][0]["args"]["gpu"] is True
    assert body["ops"][0]["args"]["expose"][0]["port"] == 8000


def test_vace_graph_connects_masks_and_trims_reference_latent(profile):
    graph = build_graph(profile, "a" * 32 + "-0", "b" * 64)
    assert graph["7"]["inputs"]["control_masks"] == ["6", 1]
    assert graph["10"]["inputs"]["trim_amount"] == ["7", 3]
    assert graph["11"]["inputs"]["samples"] == ["10", 0]


@pytest.mark.asyncio
async def test_provisioning_uses_single_replica_endpoints(tmp_path, profile, signing_key):
    created = []

    def handle(request):
        if request.url.path == "/deployments/create":
            body = json.loads(request.content)
            created.append(body)
            return httpx.Response(200, json={"id": f"deployment-{len(created)}"})
        if request.method == "GET":
            deployment_id = request.url.path.split("/")[-1]
            return httpx.Response(200, json={"status": "RUNNING", "endpoints": [{"port": 8000, "url": f"https://{deployment_id}.example", "online": True}]})
        return httpx.Response(200, json={"status": "STOPPING"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="https://api.nosana.com") as http:
        api = NosanaAPI(client=http, api_key="test-only")
        result = await provision(tmp_path / "pool.json", image="example/worker", market="a" * 44,
                                 count=2, profile=profile, public_key=public_key(signing_key), api=api)
    assert len(created) == 2
    assert all(body["replicas"] == 1 for body in created)
    assert len({entry["url"] for entry in result["deployments"]}) == 2
