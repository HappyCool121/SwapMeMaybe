from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


class FFmpegOutput:
    """Two bounded pipe writers keep raw video and audio in the same FFmpeg clock."""

    def __init__(self, destination: str, codec: str = "auto"):
        self.video_queue, self.audio_queue = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
        self.error = None
        self.is_live = destination.startswith(("srt:", "tcp:"))
        parsed = urlsplit(destination)
        self.receiver_url = f"{parsed.scheme}://{parsed.netloc}" if self.is_live else None
        self.connected = threading.Event()
        self.lines = deque(maxlen=30)
        video_read, self.video_write = os.pipe()
        audio_read, self.audio_write = os.pipe()
        if codec == "auto":
            codec = "h264_videotoolbox" if platform.system() == "Darwin" and self.is_live else "libx264"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-progress", "pipe:1", "-stats_period", "0.1",
                   "-thread_queue_size", "4", "-probesize", "32", "-analyzeduration", "0",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720", "-framerate", "24", "-i", f"pipe:{video_read}",
                   "-thread_queue_size", "4", "-probesize", "32", "-analyzeduration", "0",
                   "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", f"pipe:{audio_read}",
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", codec, "-pix_fmt", "yuv420p",
                   "-b:v", "4M", "-bf", "0", "-g", "48", "-c:a", "aac", "-b:a", "128k"]
        if codec == "libx264":
            command.extend(["-preset", "veryfast", "-tune", "zerolatency"])
        if self.is_live:
            command.extend(["-muxdelay", "0", "-muxpreload", "0", "-flush_packets", "1", "-f", "mpegts", destination])
        else:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            command.extend(["-movflags", "+faststart", destination])
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, pass_fds=(video_read, audio_read))
        except BaseException:
            for fd in (video_read, audio_read, self.video_write, self.audio_write):
                os.close(fd)
            raise
        os.close(video_read)
        os.close(audio_read)
        self.threads = [threading.Thread(target=self._write, args=(self.video_queue, self.video_write), daemon=True),
                        threading.Thread(target=self._write, args=(self.audio_queue, self.audio_write), daemon=True)]
        for thread in self.threads:
            thread.start()
        self.logger = threading.Thread(target=self._logs, daemon=True)
        self.logger.start()
        self.progress = threading.Thread(target=self._progress, daemon=True)
        self.progress.start()

    def _logs(self):
        for line in iter(self.process.stderr.readline, b""):
            self.lines.append(line.decode(errors="replace").strip())

    def _progress(self):
        for line in iter(self.process.stdout.readline, b""):
            if line.startswith(b"frame="):
                try:
                    if int(line.split(b"=", 1)[1]) > 0:
                        self.connected.set()
                except ValueError:
                    pass

    def prime(self, stop: threading.Event, timeout: float = 30):
        """Open the stream listener using bounded slate/audio preroll before the camera clock starts."""
        if not self.is_live:
            return
        deadline = time.monotonic() + timeout
        video = np.full((720, 1280, 3), 24, np.uint8).tobytes()
        audio = np.zeros(2000, dtype="<f4").tobytes()
        while not self.connected.is_set():
            for pending, payload in ((self.video_queue, video), (self.audio_queue, audio)):
                while True:
                    if stop.is_set():
                        raise RuntimeError("session stopped while waiting for OBS")
                    if self.error or self.process.poll() is not None:
                        raise RuntimeError("FFmpeg stopped: " + " | ".join(self.lines))
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"connect OBS to {self.receiver_url} within 30 seconds")
                    try:
                        pending.put(payload, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            time.sleep(1 / 24)

    def _write(self, pending, descriptor):
        try:
            while True:
                data = pending.get()
                if data is None:
                    break
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        except (BrokenPipeError, OSError) as exc:
            self.error = str(exc)
        finally:
            os.close(descriptor)

    def send(self, rgb: np.ndarray, audio: np.ndarray):
        if self.error or self.process.poll() is not None:
            raise RuntimeError("FFmpeg stopped: " + " | ".join(self.lines))
        if rgb.shape != (720, 1280, 3) or audio.shape != (2000,):
            raise ValueError("output must be 720p RGB with 2000 mono 48 kHz samples per video frame")
        try:
            self.video_queue.put(rgb.astype(np.uint8, copy=False).tobytes(), timeout=0.25)
            self.audio_queue.put(audio.astype("<f4", copy=False).tobytes(), timeout=0.25)
        except queue.Full as exc:
            raise RuntimeError("output is not consuming frames; connect OBS or check encoder speed") from exc

    def close(self):
        for pending in (self.video_queue, self.audio_queue):
            try:
                pending.put(None, timeout=1)
            except queue.Full:
                self.process.terminate()
        for thread in self.threads:
            thread.join(timeout=3)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.logger.join(timeout=1)
        self.progress.join(timeout=1)
        self.process.stderr.close()
        self.process.stdout.close()
