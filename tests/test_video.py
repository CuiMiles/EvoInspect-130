from __future__ import annotations

from evoinspect.video import StablePresenceTracker, VideoDetectorConfig


def test_presence_tracker_emits_appearance_and_repeat_only_after_stability() -> None:
    tracker = StablePresenceTracker(("bottle", "cup"), stable_samples=2)
    assert tracker.update({"bottle": True})[0] == ()
    assert tracker.update({"bottle": True}) == (("bottle",), ("bottle",))
    assert tracker.update({"bottle": True})[0] == ()
    tracker.update({"bottle": False})
    assert tracker.update({"bottle": False})[1] == ()
    tracker.update({"bottle": True})
    assert tracker.update({"bottle": True})[0] == ("bottle",)


def test_video_config_rejects_duplicate_steps() -> None:
    try:
        VideoDetectorConfig(expected_steps=("cup", "cup"))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate sequence should be rejected")
