from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize_frame(bgr: np.ndarray) -> np.ndarray:
    """Fit the camera into 720p without changing the person's aspect ratio."""
    import cv2
    height, width = bgr.shape[:2]
    scale = min(1280 / width, 720 / height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    fitted = cv2.cvtColor(cv2.resize(bgr, target), cv2.COLOR_BGR2RGB)
    output = np.full((720, 1280, 3), 24, np.uint8)
    x, y = (1280 - target[0]) // 2, (720 - target[1]) // 2
    output[y:y + target[1], x:x + target[0]] = fitted
    return output


class SyntheticSource:
    def read(self, frame_id: int) -> np.ndarray:
        image = Image.new("RGB", (1280, 720), (45, 55, 68))
        draw = ImageDraw.Draw(image)
        draw.ellipse((535, 65, 745, 255), fill=(193, 153, 127))
        draw.rounded_rectangle((426, 240, 854, 600), radius=30, fill=(51, 128, 170))
        draw.text((30, 30), f"SYNTHETIC TRANSPORT TEST — frame {frame_id}", fill="white", font_size=24)
        draw.line((426, 330 + (frame_id % 80), 854, 330 + (frame_id % 80)), fill=(75, 150, 195), width=8)
        return np.asarray(image).copy()

    def close(self):
        pass


class FileSource:
    def __init__(self, path: Path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24
        self.position, self.last = -1, None

    def read(self, frame_id: int) -> np.ndarray | None:
        target = int(frame_id * self.fps / 24)
        while self.position < target:
            success, bgr = self.cap.read()
            if not success:
                return None
            self.position += 1
            self.last = normalize_frame(bgr)
        return self.last.copy() if self.last is not None else None

    def close(self):
        self.cap.release()


class CameraSource:
    """Continuously drain camera frames so a slower consumer cannot accumulate camera latency."""

    def __init__(self, index: int):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open camera; check macOS Camera permission and device index")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.latest, self.timestamp = None, 0.0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while not self.stopping.is_set():
            success, bgr = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue
            rgb = normalize_frame(bgr)
            with self.lock:
                self.latest, self.timestamp = rgb, time.monotonic()

    def read(self, frame_id: int) -> np.ndarray | None:
        with self.lock:
            if self.latest is None or time.monotonic() - self.timestamp > 0.25:
                return None
            return self.latest.copy()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2)
        self.cap.release()


class AudioBuffer:
    def __init__(self, seconds: int = 16):
        self.limit = seconds * 48_000
        self.chunks: deque[tuple[int, np.ndarray]] = deque()
        self.lock = threading.Lock()

    def add(self, start_sample: int, samples: np.ndarray):
        samples = np.asarray(samples, np.float32).reshape(-1).copy()
        with self.lock:
            self.chunks.append((start_sample, samples))
            cutoff = start_sample + len(samples) - self.limit
            while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) <= cutoff:
                self.chunks.popleft()

    def read(self, start_sample: int, count: int) -> np.ndarray:
        result = np.zeros(count, np.float32)
        with self.lock:
            for begin, samples in self.chunks:
                left, right = max(begin, start_sample), min(begin + len(samples), start_sample + count)
                if left < right:
                    result[left - start_sample:right - start_sample] = samples[left - begin:right - begin]
            while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) <= start_sample:
                self.chunks.popleft()
        return result


class Microphone:
    def __init__(self, buffer: AudioBuffer, epoch: float, device: int | None):
        import sounddevice as sd
        self.buffer, self.epoch = buffer, epoch
        self.status_count = 0
        self.offset = None

        def callback(data, frames, callback_time, status):
            if status:
                self.status_count += 1
            # PortAudio's ADC timestamp identifies when these samples were captured.
            if self.offset is None:
                self.offset = time.monotonic() - callback_time.currentTime
            captured = callback_time.inputBufferAdcTime + self.offset - self.epoch
            self.buffer.add(round(captured * 48_000), data[:, 0])

        self.stream = sd.InputStream(samplerate=48_000, channels=1, dtype="float32", device=device,
                                     blocksize=0, callback=callback)
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()
