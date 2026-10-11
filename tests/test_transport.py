import io
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from liveswap.archive import pack_result, pack_window, unpack_result, unpack_window
from liveswap.auth import Verifier, public_key, sign
from liveswap.contracts import Result


def test_frame_and_white_mask_survive_transport(profile, inputs):
    window, frames, masks, _ = inputs
    parsed, rgb, decoded_masks = unpack_window(pack_window(window, frames, masks, profile), profile)
    assert parsed == window
    assert np.array_equal(decoded_masks[0], masks[0])
    assert decoded_masks[0][40, 50] == 255
    assert decoded_masks[0][0, 0] == 0
    assert np.abs(rgb[0].astype(int) - frames[0]).max() <= 2


def test_unexpected_zip_members_rejected(profile, inputs):
    window, frames, masks, _ = inputs
    stream = io.BytesIO(pack_window(window, frames, masks, profile))
    with zipfile.ZipFile(stream, "a") as archive:
        archive.writestr("../../private.pem", "no")
    with pytest.raises(ValueError, match="members"):
        unpack_window(stream.getvalue(), profile)


def test_compressed_zip_bomb_rejected(monkeypatch, profile, inputs):
    from liveswap import archive as module
    monkeypatch.setattr(module, "MAX_UNPACKED", 100)
    window, frames, masks, _ = inputs
    with pytest.raises(ValueError, match="unpacked"):
        unpack_window(pack_window(window, frames, masks, profile), profile)


def test_result_must_preserve_session_and_frame_sequence(profile, inputs):
    window, frames, _, _ = inputs
    result = Result(window=window, backend="mock", output_ids=window.output_ids, processing_seconds=0.1)
    output = [frames[0]] * profile.output_length
    parsed, images = unpack_result(pack_result(result, output), profile)
    assert parsed.window == window
    assert len(images) == 9


def test_signed_requests_cannot_be_replayed_or_modified(signing_key):
    verifier = Verifier(public_key(signing_key))
    headers = sign(signing_key, "POST", "/jobs", b"frames")
    verifier.verify("POST", "/jobs", b"frames", headers)
    with pytest.raises(ValueError):
        verifier.verify("POST", "/jobs", b"frames", headers)
    with pytest.raises(ValueError):
        Verifier(public_key(signing_key)).verify("POST", "/jobs", b"changed", headers)


def test_request_signature_expires(signing_key):
    headers = sign(signing_key, "GET", "/readyz")
    with pytest.raises(ValueError):
        Verifier(public_key(signing_key)).verify("GET", "/readyz", b"", headers, now=int(time.time()) + 61)


def test_transparent_reference_is_flattened_to_neutral_background():
    from liveswap.archive import decode_image
    stream = io.BytesIO()
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    image.putpixel((1, 1), (30, 80, 160, 255))
    image.save(stream, format="PNG")
    decoded = decode_image(stream.getvalue())
    assert np.array_equal(decoded[0, 0], [127, 127, 127])
    assert np.array_equal(decoded[1, 1], [30, 80, 160])
