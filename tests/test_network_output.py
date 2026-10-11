"""Opt-in actual FFmpeg/TCP smoke check: LIVESWAP_INTEGRATION=1 pytest -q."""

import json
import os
import shutil
import socket
import subprocess
import threading
import time

import numpy as np
import pytest

from liveswap.client.output import FFmpegOutput


@pytest.mark.skipif(os.environ.get("LIVESWAP_INTEGRATION") != "1" or not shutil.which("ffmpeg"),
                    reason="set LIVESWAP_INTEGRATION=1 to run the actual localhost encoder/receiver test")
def test_tcp_carries_matching_video_and_audio(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    target = tmp_path / "received.mkv"
    errors = []
    stop = threading.Event()

    def produce():
        output = FFmpegOutput(f"tcp://127.0.0.1:{port}?listen=1", codec="libx264")
        try:
            output.prime(stop, timeout=10)
            start = time.monotonic()
            for index in range(48):
                time.sleep(max(0, start + index / 24 - time.monotonic()))
                output.send(np.full((720, 1280, 3), 80, np.uint8), np.zeros(2000, np.float32))
        except BaseException as exc:
            errors.append(exc)
        finally:
            output.close()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    time.sleep(0.8)
    receiver = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                 "-f", "mpegts", "-i", f"tcp://127.0.0.1:{port}", "-c", "copy", str(target)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, error = receiver.communicate(timeout=15)
    except BaseException:
        stop.set()
        receiver.kill()
        receiver.communicate()
        raise
    finally:
        producer.join(timeout=15)
    assert not producer.is_alive()
    assert not errors, errors
    assert receiver.returncode == 0, error.decode()
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(target)]))
    video, audio = data["streams"]
    assert (video["width"], video["height"], video["r_frame_rate"]) == (1280, 720, "24/1")
    assert audio["sample_rate"] == "48000"
    assert abs(float(video["start_time"]) - float(audio["start_time"])) < 0.05
    assert float(data["format"]["duration"]) >= 2
