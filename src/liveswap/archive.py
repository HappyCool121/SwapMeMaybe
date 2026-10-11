from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from liveswap.contracts import Profile, Result, Window

MAX_ARCHIVE = 96 * 1024 * 1024
MAX_UNPACKED = 256 * 1024 * 1024


def prepare_garment(path: Path) -> bytes:
    """Validate locally and strip metadata before any reference is sent to a GPU host."""
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("garment image exceeds 10 MiB")
    rgb = decode_image(path.read_bytes())
    image = Image.fromarray(rgb)
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def encode_image(array: np.ndarray, *, mask: bool = False, quality: int = 94) -> bytes:
    out = io.BytesIO()
    image = Image.fromarray(array.astype(np.uint8))
    image.save(out, format="PNG" if mask else "JPEG", **({} if mask else {"quality": quality}))
    return out.getvalue()


def decode_image(data: bytes, *, mask: bool = False, size: tuple[int, int] | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("only JPEG and PNG images are supported")
            if image.width * image.height > 4096 * 4096:
                raise ValueError("image dimensions exceed the allowed limit")
            if size is not None and image.size != size:
                raise ValueError("frame or mask dimensions do not match the profile")
            image.load()
            if not mask and "A" in image.getbands():
                background = Image.new("RGBA", image.size, (127, 127, 127, 255))
                image = Image.alpha_composite(background, image.convert("RGBA"))
            return np.asarray(image.convert("L" if mask else "RGB")).copy()


def _open(data: bytes) -> zipfile.ZipFile:
    if len(data) > MAX_ARCHIVE:
        raise ValueError("archive exceeds upload limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        items = archive.infolist()
        names = [item.filename for item in items]
        if len(items) > 170 or len(names) != len(set(names)):
            raise ValueError("too many or duplicate archive members")
        if sum(item.file_size for item in items) > MAX_UNPACKED:
            raise ValueError("unpacked archive exceeds limit")
        if any(item.flag_bits & 1 for item in items):
            raise ValueError("encrypted archives are not supported")
        return archive
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("invalid ZIP archive") from exc


def pack_window(window: Window, frames: list[np.ndarray], masks: list[np.ndarray], profile: Profile) -> bytes:
    if not (len(frames) == len(masks) == profile.length == len(window.frame_ids)):
        raise ValueError("window frame/mask count mismatch")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", window.model_dump_json())
        for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
            archive.writestr(f"frames/{index:03}.jpg", encode_image(frame, quality=profile.jpeg_quality))
            archive.writestr(f"masks/{index:03}.png", encode_image(mask, mask=True))
    return stream.getvalue()


def unpack_window(data: bytes, profile: Profile) -> tuple[Window, list[np.ndarray], list[np.ndarray]]:
    with _open(data) as archive:
        window = Window.model_validate_json(archive.read("manifest.json"))
        if window.profile_id != profile.id or len(window.frame_ids) != profile.length:
            raise ValueError("incompatible processing profile")
        expected = {"manifest.json"}
        expected |= {f"frames/{i:03}.jpg" for i in range(profile.length)}
        expected |= {f"masks/{i:03}.png" for i in range(profile.length)}
        if set(archive.namelist()) != expected:
            raise ValueError("archive members must match the frame manifest exactly")
        size = (profile.width, profile.height)
        frames = [decode_image(archive.read(f"frames/{i:03}.jpg"), size=size) for i in range(profile.length)]
        masks = [decode_image(archive.read(f"masks/{i:03}.png"), mask=True, size=size) for i in range(profile.length)]
        return window, frames, masks


def pack_result(result: Result, frames: list[np.ndarray], quality: int = 94) -> bytes:
    if len(frames) != len(result.output_ids):
        raise ValueError("result frame count mismatch")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", result.model_dump_json())
        for index, frame in enumerate(frames):
            archive.writestr(f"frames/{index:03}.jpg", encode_image(frame, quality=quality))
    return stream.getvalue()


def unpack_result(data: bytes, profile: Profile) -> tuple[Result, list[np.ndarray]]:
    with _open(data) as archive:
        result = Result.model_validate_json(archive.read("manifest.json"))
        if result.window.profile_id != profile.id or len(result.output_ids) != profile.output_length:
            raise ValueError("result profile or frame count mismatch")
        if result.backend != profile.backend:
            raise ValueError("worker backend mismatch")
        expected = {"manifest.json"} | {f"frames/{i:03}.jpg" for i in range(profile.output_length)}
        if set(archive.namelist()) != expected:
            raise ValueError("unexpected result archive members")
        frames = [decode_image(archive.read(f"frames/{i:03}.jpg"), size=(profile.width, profile.height)) for i in range(profile.output_length)]
    return result, frames
