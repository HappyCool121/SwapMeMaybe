from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Profile(StrictModel):
    name: str = "vace-384-17"
    backend: Literal["comfy", "mock"] = "comfy"
    width: int = Field(default=384, ge=128, le=1024, multiple_of=16)
    height: int = Field(default=384, ge=128, le=1024, multiple_of=16)
    length: int = Field(default=17, ge=5, le=81)
    overlap: int = Field(default=4, ge=1)
    generation_fps: Literal[12] = 12
    output_fps: Literal[24] = 24
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=5.0, ge=1, le=20)
    shift: float = Field(default=8.0, ge=0.1, le=30)
    sampler: str = "uni_pc"
    scheduler: str = "simple"
    seed: int = Field(default=0, ge=0)
    model: str = "wan2.1_vace_1.3B_fp16.safetensors"
    text_encoder: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae: str = "wan_2.1_vae.safetensors"
    weight_dtype: Literal["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"] = "default"
    lora: str | None = None
    lora_strength: float = Field(default=1.0, ge=0, le=2)
    acceleration_verified: bool = False
    prompt: str = (
        "A realistic live camera recording. The person wears the top shown in the reference image. "
        "Preserve its color, fabric, neckline and fit. Natural folds and lighting, stable garment texture."
    )
    negative_prompt: str = "flicker, changing clothing, distorted hands, distorted face, text, watermark"
    jpeg_quality: int = Field(default=94, ge=80, le=100)

    @model_validator(mode="after")
    def valid_temporal_shape(self) -> Profile:
        if (self.length - 1) % 4 or self.overlap >= self.length:
            raise ValueError("length must be 4n+1 and overlap must be smaller than length")
        if self.backend == "comfy" and self.steps <= 6 and not (self.lora and self.acceleration_verified):
            raise ValueError("4–6 step profiles require an explicitly verified compatible LoRA")
        for name in (self.model, self.text_encoder, self.vae, self.lora):
            if name and (Path(name).name != name or name in (".", "..")):
                raise ValueError("model names must be basenames inside ComfyUI's model directories")
        return self

    @property
    def id(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def stride(self) -> int:
        return self.length - self.overlap

    @property
    def output_length(self) -> int:
        return 2 * self.length - 1

    @classmethod
    def load(cls, path: Path) -> Profile:
        return cls.model_validate_json(path.read_text())


class Crop(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)


class Window(StrictModel):
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    window_id: int = Field(ge=0)
    profile_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    garment_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    # Frame identifiers refer to the 24 fps source timeline, not worker clocks.
    frame_ids: list[int] = Field(min_length=5, max_length=81)
    pts_us: list[int] = Field(min_length=5, max_length=81)
    crop: Crop
    budget_ms: int = Field(gt=0, le=120_000)

    @model_validator(mode="after")
    def aligned(self) -> Window:
        if len(self.frame_ids) != len(self.pts_us):
            raise ValueError("frame identifiers and timestamps must have matching lengths")
        for i, (frame_id, pts) in enumerate(zip(self.frame_ids, self.pts_us, strict=True)):
            if frame_id < 0 or pts != frame_pts(frame_id):
                raise ValueError("timestamps must match the normalized 24 fps timeline")
            if i and frame_id != self.frame_ids[i - 1] + 2:
                raise ValueError("generation frames must be consecutive 12 fps samples")
        return self

    @property
    def job_id(self) -> str:
        return f"{self.session_id}-{self.window_id}"

    @property
    def output_ids(self) -> list[int]:
        return list(range(self.frame_ids[0], self.frame_ids[-1] + 1))


class Result(StrictModel):
    window: Window
    backend: Literal["comfy", "mock"]
    output_ids: list[int]
    processing_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def complete(self) -> Result:
        if self.output_ids != self.window.output_ids:
            raise ValueError("result must contain every 24 fps timestamp in the source interval")
        return self


def frame_pts(frame_id: int) -> int:
    return round(frame_id * 1_000_000 / 24)


def capacity(profile: Profile, turnaround_p95: float, workers: int = 1, margin: float = 1.0) -> dict:
    if turnaround_p95 <= 0 or not math.isfinite(turnaround_p95) or workers < 1:
        raise ValueError("turnaround must be finite and positive; workers must be positive")
    required = (profile.length - 1) / 12 + turnaround_p95 + margin
    delay = math.ceil(max(5, required) * 2) / 2
    return {
        "turnaround_p95_seconds": turnaround_p95,
        "useful_fps_per_worker": profile.stride / turnaround_p95,
        "pool_useful_fps": workers * profile.stride / turnaround_p95,
        "workers_with_25_percent_headroom": math.ceil(15 * turnaround_p95 / profile.stride),
        "required_fixed_delay_seconds": delay,
        "delay_pass": delay <= 10,
        "throughput_pass": workers * profile.stride / turnaround_p95 >= 15,
    }
