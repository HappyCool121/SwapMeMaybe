from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import typer

from liveswap.auth import create_key, load_key, public_key
from liveswap.contracts import Profile, capacity

app = typer.Typer(help="Clothing replacement streams with a fixed delay and warm Nosana GPU workers.", no_args_is_help=True)
nosana_app = typer.Typer(help="Render, provision, inspect, and stop single-replica worker deployments.")
app.add_typer(nosana_app, name="nosana")


def print_json(value):
    typer.echo(json.dumps(value, indent=2))


@app.command()
def keygen(out: Path = Path(".liveswap/client.pem")):
    """Create a private client key (0600); print only its public verification key."""
    typer.echo(create_key(out))


@app.command("public-key")
def public(key: Path = Path(".liveswap/client.pem")):
    """Print the public half of an existing signing key."""
    typer.echo(public_key(load_key(key)))


@app.command("prepare-models")
def models(out: Path = Path("models/mediapipe")):
    """Download the two versioned MediaPipe client models and record checksums."""
    from liveswap.assets import prepare_models
    print_json(prepare_models(out))


@app.command()
def worker(profile: Path = Path("profiles/mock.json"), public: str = typer.Option(..., "--public-key", envvar="LIVESWAP_PUBLIC_KEY"),
           port: int = 8000, data_dir: Path = Path(".liveswap/worker"), mock_delay: float = 0):
    """Serve a local worker. Docker's entrypoint starts ComfyUI for production."""
    import uvicorn

    from liveswap.worker.app import Settings, create_app
    settings = Settings.from_env() if "LIVESWAP_PROFILE_JSON" in os.environ else Settings(
        profile=Profile.load(profile), public_key=public, data_dir=data_dir,
        mock_delay=mock_delay, exit_on_idle=False,
        comfy_url=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"),
        rife_repo=Path(os.environ.get("RIFE_REPO", "/opt/rife")),
        rife_weights=Path(os.environ.get("RIFE_WEIGHTS", "/models/rife")),
    )
    uvicorn.run(create_app(settings), host="127.0.0.1", port=port, access_log=False)


@app.command("capacity")
def show_capacity(turnaround: float = typer.Option(..., help="Measured warm p95 window turnaround in seconds."),
                  workers: int = 1, profile: Path = Path("profiles/vace.json")):
    """Calculate useful throughput and required fixed delay, including overlap."""
    print_json(capacity(Profile.load(profile), turnaround, workers))


@app.command("matrix-profiles")
def matrix(out: Path = Path("artifacts/profiles"), profile: Path = Path("profiles/vace.json")):
    """Generate the eight resolution/window benchmark candidates."""
    original = Profile.load(profile)
    out.mkdir(parents=True, exist_ok=True)
    for side in (384, 512):
        for length, overlap in ((17, 4), (25, 4), (33, 8), (81, 16)):
            candidate = Profile.model_validate({**original.model_dump(), "name": f"vace-{side}-{length}",
                                                "width": side, "height": side, "length": length, "overlap": overlap})
            target = out / f"{candidate.name}.json"
            target.write_text(candidate.model_dump_json(indent=2) + "\n")
            typer.echo(str(target))


@app.command()
def benchmark(worker: str = typer.Option(..., "--worker"), garment: Path = typer.Option(...),
              profile: Path = Path("profiles/vace.json"), key: Path = Path(".liveswap/client.pem"),
              video: Path | None = None, synthetic: bool = False, samples: int = 10,
              report: Path = Path("artifacts/benchmark.json"), model_dir: Path = Path("models/mediapipe"),
              hourly_price: float | None = None):
    """Measure complete warm window turnaround; this does not certify appearance."""
    from liveswap.benchmark import benchmark as measure
    result = asyncio.run(measure(url=worker, key=key, profile=Profile.load(profile), garment=garment,
                                video=video, synthetic=synthetic, samples=samples, report_path=report,
                                model_dir=model_dir, hourly_price=hourly_price))
    print_json(result)


@app.command("record-visual-review")
def visual_review(report: Path, accepted_clips: int, reviewed_clips: int = 20):
    """Record YOUR completed visual review; never evaluates or fabricates image quality."""
    data = json.loads(report.read_text())
    if data.get("synthetic") or data.get("backend") != "comfy":
        raise typer.BadParameter("synthetic/passthrough output cannot qualify for visual review")
    if reviewed_clips < 20 or not 0 <= accepted_clips <= reviewed_clips:
        raise typer.BadParameter("review at least 20 clips and provide a valid accepted count")
    data["visual_review_pass"] = accepted_clips / reviewed_clips >= 0.9
    data["visual_review"] = {"accepted_clips": accepted_clips, "reviewed_clips": reviewed_clips,
                             "recorded_by_operator": True}
    report.write_text(json.dumps(data, indent=2) + "\n")
    print_json(data["visual_review"])


@app.command()
def run(garment: Path | None = None, worker: list[str] = typer.Option([], "--worker"),
        deployment: Path | None = None, profile: Path = Path("profiles/vace.json"),
        key: Path = Path(".liveswap/client.pem"), model_dir: Path = Path("models/mediapipe"),
        camera: int = 0, microphone: int | None = None, no_microphone: bool = False,
        microphone_delay_ms: float = 0,
        video: Path | None = None, passthrough: bool = False,
        delay: float = 8, duration: float = 0,
        output: str = "tcp://127.0.0.1:10000?listen=1", codec: str = "auto",
        benchmark: Path | None = None, experimental: bool = False,
        metrics: Path = Path("artifacts/session.json")):
    """Stream camera+microphone or representative video; duration 0 means unlimited."""
    from liveswap.client.engine import ClientOptions, run_managed
    options = ClientOptions(profile=Profile.load(profile), worker_urls=worker, key_path=key, garment_path=garment,
                            model_dir=model_dir, camera=camera, microphone=microphone, no_microphone=no_microphone,
                            microphone_delay_ms=microphone_delay_ms,
                            video=video, passthrough=passthrough, delay=delay, duration=duration, output=output,
                            codec=codec, benchmark=benchmark, experimental=experimental, metrics_path=metrics)
    result = asyncio.run(run_managed(options, deployment=deployment, status=lambda value: print_json(value)))
    print_json(result)


@app.command()
def gui():
    """Open the desktop client (install the client extra first)."""
    from liveswap.client.gui import launch
    launch()


@app.command()
def devices():
    """List microphone devices without starting capture. Camera selection uses its index."""
    import sounddevice
    typer.echo(str(sounddevice.query_devices()))


@app.command()
def doctor():
    """Inspect runtime and encoder/transport availability without opening devices."""
    import platform
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    protocols = subprocess.check_output([ffmpeg, "-hide_banner", "-protocols"], text=True) if ffmpeg else ""
    encoders = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], text=True) if ffmpeg else ""
    print_json({"python": platform.python_version(), "platform": platform.platform(),
                "ffmpeg": ffmpeg, "tcp": "tcp" in protocols.split(), "srt": "srt" in protocols.split(),
                "videotoolbox": "h264_videotoolbox" in encoders, "libx264": "libx264" in encoders,
                "mediapipe_models": all((Path("models/mediapipe") / name).exists() for name in ("pose_landmarker_lite.task", "selfie_multiclass.tflite"))})


@app.command()
def demo(duration: float = 6, delay: float = 5, out: Path = Path("artifacts/demo.mp4"), mock_delay: float = 0):
    """Run a real local HTTP/client/encoder test with SYNTHETIC content, not diffusion."""
    from PIL import Image

    from liveswap.client.engine import ClientOptions, Session

    if duration <= 0:
        raise typer.BadParameter("demo duration must be positive")
    with tempfile.TemporaryDirectory(prefix="liveswap-demo-") as temporary:
        folder = Path(temporary)
        key_path = folder / "client.pem"
        public = create_key(key_path)
        garment = folder / "garment.jpg"
        Image.new("RGB", (384, 384), (40, 100, 160)).save(garment)
        profile = Profile(name="mock-384-17", backend="mock", steps=4, cfg=1)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {**os.environ, "LIVESWAP_PUBLIC_KEY": public, "LIVESWAP_PROFILE_JSON": profile.model_dump_json(),
               "LIVESWAP_DATA_DIR": str(folder / "worker"), "LIVESWAP_EXIT_ON_IDLE": "0", "LIVESWAP_MOCK_DELAY": str(mock_delay)}
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix(".worker.log").open("w") as log:
            server = subprocess.Popen([sys.executable, "-m", "uvicorn", "liveswap.worker.app:create_app", "--factory",
                                       "--host", "127.0.0.1", "--port", str(port), "--no-access-log"], env=env, stdout=log, stderr=log)
            try:
                options = ClientOptions(profile=profile, worker_urls=[f"http://127.0.0.1:{port}"], key_path=key_path,
                                        garment_path=garment, synthetic=True, no_microphone=True, delay=delay,
                                        duration=duration, output=str(out), codec="libx264", metrics_path=out.with_suffix(".json"))
                print_json(asyncio.run(Session(options).run()))
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    typer.echo(f"Synthetic transport test written to {out}; this does not demonstrate garment replacement.")


@nosana_app.command("job")
def render_job(image: str, profile: Path = Path("profiles/vace.json"), key: Path = Path(".liveswap/client.pem"),
               out: Path = Path("artifacts/nosana-job.json"), vram: int = 24):
    """Render public job JSON locally; does not spend credits."""
    from liveswap.nosana import job_definition
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(job_definition(image, Profile.load(profile), public_key(load_key(key)), vram), indent=2) + "\n")
    typer.echo(str(out))


@nosana_app.command("deploy")
def deploy(image: str, market: str, count: int = 1, profile: Path = Path("profiles/vace.json"),
           key: Path = Path(".liveswap/client.pem"), out: Path = Path(".liveswap/deployment.json")):
    """Create/start paid GPU workers using NOSANA_API_KEY; save every deployment ID."""
    from liveswap.nosana import provision
    typer.echo(f"Provisioning {count} paid Nosana worker(s). Deployment IDs will be saved to {out}.")
    print_json(asyncio.run(provision(out, image=image, market=market, count=count, profile=Profile.load(profile),
                                    public_key=public_key(load_key(key)))))


@nosana_app.command("stop")
def stop_pool(deployment: Path = Path(".liveswap/deployment.json")):
    """Stop all workers recorded in the deployment file."""
    from liveswap.nosana import NosanaPool

    async def action():
        pool = NosanaPool(deployment)
        try:
            await pool.stop()
        finally:
            await pool.close()
    asyncio.run(action())
    typer.echo("Stop requested for every recorded deployment; verify status with nosana status.")


@nosana_app.command("status")
def pool_status(deployment: Path = Path(".liveswap/deployment.json")):
    """Read recorded deployment status and public endpoint URLs."""
    from liveswap.nosana import NosanaPool

    async def action():
        pool = NosanaPool(deployment)
        try:
            return [await pool.api.request("GET", f"/deployments/{entry['id']}") for entry in pool.data["deployments"]]
        finally:
            await pool.close()
    print_json(asyncio.run(action()))


if __name__ == "__main__":
    app()
