import uuid

import numpy as np
import pytest

from liveswap.client.capture import AudioBuffer
from liveswap.contracts import Crop, Profile, Result, Window, frame_pts
from liveswap.timeline import Fallback, FrameBuffer, SourceFrame, Stitcher, WindowAssembler


def result_for(profile, session, window_id, start):
    ids = list(range(start, start + profile.length * 2, 2))
    window = Window(session_id=session, window_id=window_id, profile_id=profile.id, garment_hash="a" * 64,
                    frame_ids=ids, pts_us=[frame_pts(i) for i in ids], crop=Crop(x=2, y=2, width=8, height=8), budget_ms=5000)
    return Result(window=window, backend="mock", output_ids=window.output_ids, processing_seconds=0.1)


def test_out_of_order_windows_blend_and_preserve_original_pixels():
    profile = Profile(name="test", backend="mock", width=128, height=128, steps=4)
    session = uuid.uuid4().hex
    stitcher = Stitcher(profile, session)
    first, second = result_for(profile, session, 0, 0), result_for(profile, session, 1, 26)
    stitcher.add(second, [np.full((128, 128, 3), 200, np.uint8)] * 33)
    stitcher.add(first, [np.zeros((128, 128, 3), np.uint8)] * 33)
    original = np.full((16, 16, 3), 37, np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    mask[4:8, 4:8] = 255
    output = stitcher.composite(SourceFrame(29, original, mask))
    assert np.all(output[4:8, 4:8] == 100)
    assert np.array_equal(output[mask == 0], original[mask == 0])


def test_stale_session_and_duplicate_playback_are_rejected(profile):
    session = uuid.uuid4().hex
    stitcher = Stitcher(profile, session)
    result = result_for(profile, uuid.uuid4().hex, 0, 0)
    with pytest.raises(ValueError, match="session"):
        stitcher.add(result, [np.zeros((128, 128, 3), np.uint8)] * 9)
    original = SourceFrame(0, np.zeros((16, 16, 3), np.uint8), None)
    stitcher.composite(original)
    with pytest.raises(ValueError):
        stitcher.composite(original)


def test_missing_output_holds_half_second_then_slate():
    fallback = Fallback(32, 32)
    edited = np.full((32, 32, 3), 155, np.uint8)
    fallback.render(edited)
    for _ in range(12):
        image, valid = fallback.render(None)
        assert not valid and np.array_equal(image, edited)
    image, valid = fallback.render(None)
    assert not valid and not np.array_equal(image, edited)


def test_source_buffer_is_bounded():
    buffer = FrameBuffer(max_frames=3)
    for frame_id in range(20):
        buffer.put(SourceFrame(frame_id, np.zeros((1, 1, 3), np.uint8), None))
    assert list(buffer.frames) == [17, 18, 19]
    assert buffer.pop(1) is None


def test_window_stride_and_tracking_gap(profile):
    assembler = WindowAssembler(profile)
    found = []
    for frame_id in range(18):
        frame = SourceFrame(frame_id, np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8))
        window = assembler.push(frame)
        if window:
            found.append([item.frame_id for item in window])
    assert found == [[0, 2, 4, 6, 8], [8, 10, 12, 14, 16]]
    assembler.push(SourceFrame(18, np.zeros((2, 2, 3), np.uint8), None))
    assert not assembler.pending


def test_audio_gaps_do_not_shift_subsequent_samples():
    buffer = AudioBuffer(seconds=1)
    buffer.add(0, np.ones(2000))
    buffer.add(4000, np.full(2000, 3))
    assert np.all(buffer.read(0, 2000) == 1)
    assert np.all(buffer.read(2000, 2000) == 0)
    assert np.all(buffer.read(4000, 2000) == 3)


def test_audio_memory_is_bounded():
    buffer = AudioBuffer(seconds=1)
    for start in range(0, 96_000, 2000):
        buffer.add(start, np.ones(2000))
    assert sum(len(chunk) for _, chunk in buffer.chunks) <= 48_000
