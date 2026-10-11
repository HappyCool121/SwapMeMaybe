from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from liveswap.contracts import Crop, Profile, Result


@dataclass
class SourceFrame:
    frame_id: int
    rgb: np.ndarray
    mask: np.ndarray | None


def resize(array: np.ndarray, size: tuple[int, int], *, mask: bool = False) -> np.ndarray:
    return np.asarray(Image.fromarray(array).resize(size, Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)).copy()


def crop_for(frames: list[SourceFrame]) -> Crop:
    height, width = frames[0].rgb.shape[:2]
    boxes = []
    for frame in frames:
        if frame.mask is None:
            raise ValueError("cannot build a window without matched masks")
        ys, xs = np.nonzero(frame.mask > 16)
        if not len(xs):
            raise ValueError("empty garment mask")
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    side = min(max(width, height), math.ceil(max(x1 - x0, y1 - y0) * 1.2))
    crop_width, crop_height = min(side, width), min(side, height)
    x = max(0, min(width - crop_width, round((x0 + x1 - crop_width) / 2)))
    y = max(0, min(height - crop_height, round((y0 + y1 - crop_height) / 2)))
    return Crop(x=x, y=y, width=crop_width, height=crop_height)


def window_images(frames: list[SourceFrame], crop: Crop, profile: Profile):
    rgb, masks = [], []
    area = np.s_[crop.y:crop.y + crop.height, crop.x:crop.x + crop.width]
    for frame in frames:
        rgb.append(resize(frame.rgb[area], (profile.width, profile.height)))
        masks.append(resize(frame.mask[area], (profile.width, profile.height), mask=True))
    return rgb, masks


class WindowAssembler:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.pending: deque[SourceFrame] = deque(maxlen=profile.length)
        self.since_emit = 0
        self.emitted = False
        self.next_start = None

    def push(self, frame: SourceFrame) -> list[SourceFrame] | None:
        if frame.frame_id % 2:
            return None
        if frame.mask is None or not np.any(frame.mask):
            self.pending.clear()
            self.since_emit, self.emitted = 0, False
            return None
        if self.pending and frame.frame_id != self.pending[-1].frame_id + 2:
            self.pending.clear()
            self.since_emit, self.emitted = 0, False
        self.pending.append(frame)
        self.since_emit += 1
        if len(self.pending) == self.profile.length and (not self.emitted or self.since_emit >= self.profile.stride):
            self.emitted, self.since_emit = True, 0
            self.next_start = self.pending[0].frame_id + self.profile.stride * 2
            return list(self.pending)
        return None

    def finish(self) -> list[SourceFrame] | None:
        if not self.pending or (self.emitted and not self.since_emit):
            return None
        frames = [frame for frame in self.pending if self.next_start is None or frame.frame_id >= self.next_start]
        if not frames:
            return None
        while len(frames) < self.profile.length:
            last = frames[-1]
            frames.append(SourceFrame(last.frame_id + 2, last.rgb, last.mask))
        return frames


@dataclass
class Patch:
    crop: Crop
    rgb: np.ndarray
    weight: float


class Stitcher:
    """Bounded by the playback horizon; normalizes available overlap contributions."""

    def __init__(self, profile: Profile, session_id: str, horizon: int = 384):
        self.profile, self.session_id, self.horizon = profile, session_id, horizon
        self.patches: dict[int, dict[int, Patch]] = {}
        self.played = -1

    def add(self, result: Result, frames: list[np.ndarray]) -> None:
        window = result.window
        if window.session_id != self.session_id or window.profile_id != self.profile.id:
            raise ValueError("stale session or incompatible profile")
        if len(frames) != len(result.output_ids):
            raise ValueError("incomplete result")
        fade = 2 * self.profile.overlap - 1
        for index, (frame_id, frame) in enumerate(zip(result.output_ids, frames, strict=True)):
            if frame_id <= self.played or frame_id > self.played + self.horizon:
                continue
            weight = 1.0
            if window.window_id and index < fade:
                weight = index / max(1, fade - 1)
            if index >= len(frames) - fade:
                weight = min(weight, (len(frames) - 1 - index) / max(1, fade - 1))
            self.patches.setdefault(frame_id, {})[window.window_id] = Patch(window.crop, frame, weight)

    def composite(self, source: SourceFrame) -> np.ndarray | None:
        if source.frame_id <= self.played:
            raise ValueError("playback timestamps must increase")
        self.played = source.frame_id
        contributions = list(self.patches.pop(source.frame_id, {}).values())
        self.patches = {frame_id: value for frame_id, value in self.patches.items() if frame_id > self.played}
        if source.mask is None or not contributions:
            return None
        height, width = source.rgb.shape[:2]
        accumulator = np.zeros((height, width, 3), np.float32)
        weights = np.zeros((height, width), np.float32)
        if len(contributions) == 1 or not any(patch.weight > 0 for patch in contributions):
            for patch in contributions:
                patch.weight = 1.0
        for patch in contributions:
            crop = patch.crop
            if crop.x + crop.width > width or crop.y + crop.height > height:
                raise ValueError("result crop exceeds original frame")
            region = np.s_[crop.y:crop.y + crop.height, crop.x:crop.x + crop.width]
            accumulator[region] += resize(patch.rgb, (crop.width, crop.height)) * patch.weight
            weights[region] += patch.weight
        replacement = accumulator / np.maximum(weights[..., None], 1e-6)
        alpha = source.mask.astype(np.float32) / 255
        alpha *= weights > 0
        output = source.rgb.copy()
        edited = alpha > 0
        output[edited] = np.clip(source.rgb[edited] * (1 - alpha[edited, None]) + replacement[edited] * alpha[edited, None], 0, 255).astype(np.uint8)
        return output


class FrameBuffer:
    def __init__(self, max_frames: int = 384):
        self.max_frames = max_frames
        self.frames: OrderedDict[int, SourceFrame] = OrderedDict()

    def put(self, frame: SourceFrame):
        self.frames[frame.frame_id] = frame
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)

    def pop(self, frame_id: int) -> SourceFrame | None:
        return self.frames.pop(frame_id, None)


def slate(width: int, height: int, text: str = "Processing / reconnecting") -> np.ndarray:
    image = Image.new("RGB", (width, height), (22, 25, 31))
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, height // 2), text, fill=(230, 233, 239), anchor="mm", font_size=28)
    return np.asarray(image).copy()


class Fallback:
    def __init__(self, width: int = 1280, height: int = 720):
        self.last_good: np.ndarray | None = None
        self.missing = 0
        self.slate = slate(width, height)

    def render(self, edited: np.ndarray | None) -> tuple[np.ndarray, bool]:
        if edited is not None:
            self.last_good, self.missing = edited, 0
            return edited, True
        self.missing += 1
        if self.last_good is not None and self.missing <= 12:
            return self.last_good.copy(), False
        return self.slate.copy(), False
