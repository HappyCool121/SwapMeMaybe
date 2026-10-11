from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

MEDIAPIPE_MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "selfie_multiclass.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
}
MODEL_SHA256 = {
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    "selfie_multiclass.tflite": "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
}


def prepare_models(folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in MEDIAPIPE_MODELS.items():
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != MODEL_SHA256[name]:
            raise ValueError(f"model checksum mismatch: {path}; remove that file and download it again")
        manifest[name] = {"url": url, "sha256": digest}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
