from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.last = self.raw = self.derivative = None
        self.time = None

    def apply(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        if self.last is None:
            self.last, self.raw, self.derivative, self.time = value.copy(), value.copy(), np.zeros_like(value), timestamp
            return value.copy()
        dt = max(timestamp - self.time, 1e-6)
        velocity = (value - self.raw) / dt
        derivative_alpha = 1 / (1 + 1 / (2 * math.pi * dt))
        self.derivative += derivative_alpha * (velocity - self.derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
        alpha = 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
        self.last += alpha * (value - self.last)
        self.raw, self.time = value.copy(), timestamp
        return self.last.copy()


class MediaPipeMasker:
    def __init__(self, model_dir: Path):
        import mediapipe as mp

        from liveswap.assets import MODEL_SHA256

        for filename, expected in MODEL_SHA256.items():
            if hashlib.sha256((model_dir / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"wrong MediaPipe model: {filename}; run liveswap prepare-models")

        self.mp = mp
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(model_dir / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
        ))
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base(model_asset_path=str(model_dir / "selfie_multiclass.tflite")),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True, output_confidence_masks=False,
        ))
        self.filter = OneEuro()
        self.previous_gray = self.previous_mask = None

    def mask(self, rgb: np.ndarray, pts_us: int) -> np.ndarray | None:
        import cv2

        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp_ms = pts_us // 1000
        pose = self.pose.detect_for_video(image, timestamp_ms)
        segmented = self.segmenter.segment_for_video(image, timestamp_ms)
        if not pose.pose_landmarks or segmented.category_mask is None:
            self.previous_mask = None
            return None
        landmarks = pose.pose_landmarks[0]
        if min(landmarks[11].visibility, landmarks[12].visibility) < 0.5:
            self.previous_mask = None
            return None
        points = self.filter.apply(np.array([[landmark.x, landmark.y] for landmark in landmarks]), pts_us / 1e6)
        height, width = rgb.shape[:2]
        labels = segmented.category_mask.numpy_view().copy()
        mh, mw = labels.shape
        shoulders = points[[11, 12]]
        hips = points[[23, 24]]
        shoulder_width = max(abs(shoulders[0, 0] - shoulders[1, 0]), 0.1)
        torso_height = max(float(hips[:, 1].mean() - shoulders[:, 1].mean()), 0.2)
        x0 = max(0, int((shoulders[:, 0].min() - shoulder_width * 0.55) * mw))
        x1 = min(mw, int((shoulders[:, 0].max() + shoulder_width * 0.55) * mw))
        y0 = max(0, int((shoulders[:, 1].min() - torso_height * 0.18) * mh))
        y1 = min(mh, int((hips[:, 1].max() + torso_height * 0.05) * mh))
        allowed = np.zeros_like(labels, dtype=np.float32)
        allowed[y0:y1, x0:x1] = 1
        # The model's labels are background, hair, body skin, face skin, clothes, other.
        editable = (labels == 4).astype(np.float32) * allowed
        hands = hand_protection(points, [landmark.visibility for landmark in landmarks], mh, mw)
        editable[hands] = 0
        if editable.sum() < 100:
            self.previous_mask = None
            return None
        gray = cv2.cvtColor(cv2.resize(rgb, (mw, mh)), cv2.COLOR_RGB2GRAY)
        probability = editable
        if self.previous_mask is not None and self.previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(gray, self.previous_gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            xx, yy = np.meshgrid(np.arange(mw, dtype=np.float32), np.arange(mh, dtype=np.float32))
            aligned = cv2.remap(self.previous_mask, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            probability = 0.7 * editable + 0.3 * aligned
        probability = cv2.GaussianBlur(probability, (3, 3), 0.7) * editable
        self.previous_gray, self.previous_mask = gray, probability
        mask = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        protected = cv2.resize((labels != 4).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        protected |= cv2.resize(hands.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        mask[protected > 0] = 0
        return np.clip(mask * 255, 0, 255).astype(np.uint8)

    def close(self):
        self.pose.close()
        self.segmenter.close()


def hand_protection(points: np.ndarray, visibility: list[float], height: int, width: int) -> np.ndarray:
    """Pose-derived hand hulls supplement skin labels when hands cross the garment."""
    import cv2

    result = np.zeros((height, width), np.uint8)
    for indices in ([15, 17, 19, 21], [16, 18, 20, 22]):
        confident = [index for index in indices if visibility[index] >= 0.5]
        if len(confident) >= 3:
            pixels = np.rint(points[confident] * [width, height]).astype(np.int32)
            hull = cv2.convexHull(pixels)
            cv2.fillConvexPoly(result, hull, 1)
    return cv2.dilate(result, np.ones((3, 3), np.uint8)).astype(bool)


class SyntheticMasker:
    """Development-only garment rectangle for the generated test scene."""

    def mask(self, rgb: np.ndarray, pts_us: int) -> np.ndarray:
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        mask[height // 3:5 * height // 6, width // 3:2 * width // 3] = 255
        return mask

    def close(self):
        pass
