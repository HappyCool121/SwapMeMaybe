import hashlib
import uuid

import numpy as np
import pytest

from liveswap.archive import encode_image
from liveswap.auth import create_key, load_key
from liveswap.contracts import Crop, Profile, Window, frame_pts


@pytest.fixture
def profile():
    return Profile(name="test", backend="mock", width=128, height=128, length=5, overlap=1, steps=4, cfg=1)


@pytest.fixture
def signing_key(tmp_path):
    path = tmp_path / "client.pem"
    create_key(path)
    return load_key(path)


@pytest.fixture
def inputs(profile):
    rgb = np.full((profile.height, profile.width, 3), 100, np.uint8)
    mask = np.zeros((profile.height, profile.width), np.uint8)
    mask[20:90, 30:100] = 255
    asset = encode_image(rgb)
    window = Window(session_id=uuid.uuid4().hex, window_id=0, profile_id=profile.id,
                    garment_hash=hashlib.sha256(asset).hexdigest(), frame_ids=list(range(0, 10, 2)),
                    pts_us=[frame_pts(i) for i in range(0, 10, 2)],
                    crop=Crop(x=0, y=0, width=128, height=128), budget_ms=5000)
    return window, [rgb] * 5, [mask] * 5, asset
