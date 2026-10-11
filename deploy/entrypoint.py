"""Supervise the API and ComfyUI; expose health while model resources load."""

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from liveswap.contracts import Profile
from liveswap.nosana import WAN_REPO, WAN_REVISION, model_files


def main():
    if not os.environ.get("LIVESWAP_PUBLIC_KEY"):
        raise RuntimeError("LIVESWAP_PUBLIC_KEY is required; never put the private key in the image")
    profile = Profile.model_validate_json(os.environ["LIVESWAP_PROFILE_JSON"]) if os.environ.get("LIVESWAP_PROFILE_JSON") else Profile.load(Path("profiles/vace.json"))
    root = Path(os.environ["LIVESWAP_DATA_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    comfy = Path(os.environ.get("COMFY_DIR", "/opt/comfy"))
    source_hash = hashlib.sha256()
    source_files = list(Path("/opt/app/src").rglob("*.py"))
    source_files += list((comfy / "custom_nodes" / "liveswap_nodes").rglob("*.py"))
    source_files.append(Path(__file__))
    for file in sorted(source_files):
        source_hash.update(str(file).encode())
        source_hash.update(file.read_bytes())
    os.environ["LIVESWAP_SOURCE_SHA256"] = source_hash.hexdigest()
    os.environ.setdefault("WAN_REVISION", WAN_REVISION)
    children = []

    def stop(*args):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "liveswap.worker.app:create_app", "--factory",
                                          "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]))
        if profile.backend == "comfy":
            from huggingface_hub import hf_hub_download

            if profile.lora and not (comfy / "models" / "loras" / profile.lora).is_file():
                raise FileNotFoundError("bake the verified LoRA into deploy/loras before building this image")

            mounted = Path(os.environ.get("WAN_MODEL_ROOT", "/models/wan"))
            revision = os.environ.get("WAN_REVISION", WAN_REVISION)
            for relative, category in zip(model_files(profile), ("diffusion_models", "text_encoders", "vae"), strict=True):
                candidates = [mounted / relative, mounted / category / Path(relative).name, mounted / Path(relative).name]
                source = next((candidate for candidate in candidates if candidate.is_file()), None)
                if source is None:
                    source = Path(hf_hub_download(WAN_REPO, filename=relative, revision=revision))
                target = comfy / "models" / category / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    target.unlink()
                if not target.exists():
                    target.symlink_to(source)
            children.append(subprocess.Popen([sys.executable, str(comfy / "main.py"), "--listen", "127.0.0.1",
                                              "--port", "8188", "--output-directory", str(root / "outputs"),
                                              "--disable-auto-launch", "--cache-lru", "16"], cwd=comfy))
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        if any(child.returncode not in (None, 0, -15) for child in children):
            raise RuntimeError("worker or ComfyUI exited unexpectedly")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
