from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx

from liveswap.contracts import Profile

WAN_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
WAN_REVISION = "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d"


def model_files(profile: Profile) -> list[str]:
    return [f"split_files/diffusion_models/{profile.model}",
            f"split_files/text_encoders/{profile.text_encoder}", f"split_files/vae/{profile.vae}"]


def job_definition(image: str, profile: Profile, public_key: str, required_vram: int = 24) -> dict:
    return {
        "version": "0.1", "type": "container",
        "meta": {"trigger": "liveswap", "system_resources": {"required_vram": required_vram}},
        "ops": [{"type": "container/run", "id": "worker", "args": {
            "image": image, "gpu": True, "restart_policy": "no",
            "env": {"LIVESWAP_PUBLIC_KEY": public_key, "LIVESWAP_PROFILE_JSON": profile.model_dump_json(),
                    "LIVESWAP_DATA_DIR": "/data/liveswap", "LIVESWAP_EXIT_ON_IDLE": "1",
                    "WAN_MODEL_ROOT": "/models/wan", "WAN_REVISION": WAN_REVISION},
            "expose": [{"port": 8000, "type": "api", "health_checks": [{"type": "http", "path": "/healthz",
                         "method": "GET", "expected_status": 200, "continuous": True}]}],
            "resources": [{"type": "HF", "target": "/models/wan", "repo": WAN_REPO,
                           "revision": WAN_REVISION, "files": model_files(profile)}],
        }}],
    }


class NosanaAPI:
    def __init__(self, *, client: httpx.AsyncClient | None = None, api_key: str | None = None):
        key = api_key or os.environ.get("NOSANA_API_KEY")
        if not key:
            raise ValueError("set NOSANA_API_KEY to manage Nosana deployments")
        self.http = client or httpx.AsyncClient(base_url="https://api.nosana.com", timeout=60)
        self.owns_client = client is None
        self.headers = {"Authorization": f"Bearer {key}"}
        if os.environ.get("NOSANA_USER_ID"):
            self.headers["x-user-id"] = os.environ["NOSANA_USER_ID"]

    async def request(self, method: str, path: str, body: dict | None = None):
        response = await self.http.request(method, path, headers=self.headers, **({"json": body} if body is not None else {}))
        response.raise_for_status()
        return response.json()

    async def close(self):
        if self.owns_client:
            await self.http.aclose()


class NosanaPool:
    def __init__(self, path: Path, api: NosanaAPI | None = None):
        self.path = path
        self.data = json.loads(path.read_text())
        self.api = api or NosanaAPI()

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        temporary.replace(self.path)

    async def start_and_resolve(self, timeout: float = 900) -> list[str]:
        deadline = time.monotonic() + timeout
        for entry in self.data["deployments"]:
            record = await self.api.request("GET", f"/deployments/{entry['id']}")
            if record["status"] in {"DRAFT", "STOPPED"}:
                await self.api.request("POST", f"/deployments/{entry['id']}/start")
        while time.monotonic() < deadline:
            complete = True
            for entry in self.data["deployments"]:
                record = await self.api.request("GET", f"/deployments/{entry['id']}")
                if record["status"] in {"ERROR", "INSUFFICIENT_FUNDS", "ARCHIVED"}:
                    raise RuntimeError(f"Nosana deployment {entry['id']} is {record['status']}")
                endpoints = [endpoint for endpoint in record.get("endpoints", []) if int(endpoint["port"]) == 8000]
                if endpoints:
                    entry["url"] = endpoints[0]["url"].rstrip("/")
                else:
                    complete = False
            self.save()
            if complete:
                return [entry["url"] for entry in self.data["deployments"]]
            await asyncio.sleep(2)
        raise TimeoutError("Nosana did not expose all worker endpoints within startup timeout")

    async def stop(self):
        errors = []
        for entry in self.data["deployments"]:
            try:
                await self.api.request("POST", f"/deployments/{entry['id']}/stop")
                entry["stop_requested"] = True
            except Exception as exc:
                errors.append(f"{entry['id']}: {type(exc).__name__}")
        self.save()
        if errors:
            raise RuntimeError("could not stop all deployments; run nosana stop again: " + ", ".join(errors))

    async def maintain(self, workers):
        failures = {entry["id"]: 0 for entry in self.data["deployments"]}
        while True:
            await asyncio.sleep(5)
            for entry, worker in zip(self.data["deployments"], workers, strict=True):
                try:
                    info = await worker.ready()
                    if info.get("error"):
                        raise httpx.HTTPError("worker reported unhealthy")
                    failures[entry["id"]] = 0
                    continue
                except httpx.HTTPError:
                    failures[entry["id"]] += 1
                if failures[entry["id"]] < 3:
                    continue
                try:
                    await self.api.request("POST", f"/deployments/{entry['id']}/stop")
                    for _ in range(60):
                        record = await self.api.request("GET", f"/deployments/{entry['id']}")
                        if record["status"] == "STOPPED":
                            break
                        await asyncio.sleep(2)
                    else:
                        continue
                    await self.api.request("POST", f"/deployments/{entry['id']}/start")
                    failures[entry["id"]] = 0
                    # Resolve the stable single-replica endpoint again; never mix replicas in one service.
                    await self.start_and_resolve()
                    worker.url = entry["url"]
                except httpx.HTTPError:
                    await asyncio.sleep(5)

    async def close(self):
        await self.api.close()


async def provision(path: Path, *, image: str, market: str, count: int, profile: Profile,
                    public_key: str, api: NosanaAPI | None = None) -> dict:
    if path.exists():
        raise FileExistsError("deployment metadata already exists; use a new file or restart the existing pool")
    if not 1 <= count <= 32:
        raise ValueError("worker count must be between 1 and 32")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile_id": profile.id, "public_key": public_key, "deployments": []}))
    pool = NosanaPool(path, api)
    try:
        for index in range(count):
            body = {"name": f"liveswap-{uuid.uuid4().hex[:8]}-{index}", "market": market,
                    "replicas": 1, "timeout": 60, "strategy": "SIMPLE-EXTEND", "autostart": True,
                    "job_definition": job_definition(image, profile, public_key)}
            record = await pool.api.request("POST", "/deployments/create", body)
            pool.data["deployments"].append({"id": record["id"], "url": None})
            pool.save()
        await pool.start_and_resolve()
        return pool.data
    except BaseException:
        try:
            await pool.stop()
        except Exception:
            pass  # Every created ID was durably saved for manual recovery.
        raise
    finally:
        await pool.close()
