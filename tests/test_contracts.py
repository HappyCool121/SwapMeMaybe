import math

import pytest
from pydantic import ValidationError

from liveswap.contracts import Profile, Window, capacity


def test_overlap_changes_real_capacity():
    profile = Profile(length=81, overlap=16)
    result = capacity(profile, 24.3, workers=4)
    assert result["pool_useful_fps"] == pytest.approx(10.6996, rel=0.001)
    assert result["workers_with_25_percent_headroom"] == 6
    assert not result["delay_pass"]
    assert not result["throughput_pass"]


def test_short_windows_can_pass_delay_without_claiming_throughput():
    result = capacity(Profile(), 5.1, workers=6)
    assert result["required_fixed_delay_seconds"] == 7.5
    assert result["delay_pass"] and result["throughput_pass"]


@pytest.mark.parametrize("updates", [{"length": 18}, {"overlap": 17}, {"width": 383}, {"model": "../secret"}])
def test_profile_rejects_incompatible_shapes_and_paths(updates):
    with pytest.raises(ValidationError):
        Profile(**updates)


def test_fast_profile_cannot_silently_assume_acceleration():
    with pytest.raises(ValidationError, match="verified"):
        Profile(steps=4)


@pytest.mark.parametrize("duration", [0, -1, math.inf, math.nan])
def test_capacity_rejects_unusable_measurements(profile, duration):
    with pytest.raises(ValueError):
        capacity(profile, duration)


def test_frame_timestamp_mismatch_is_rejected(inputs):
    window = inputs[0].model_dump()
    window["pts_us"][2] += 1
    with pytest.raises(ValidationError):
        Window.model_validate(window)
