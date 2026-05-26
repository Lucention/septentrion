import datetime as dt

import pytest
from pydantic import ValidationError

from septentrion.schemas import AisRecord, Detection, MatchedContact


def test_detection_requires_valid_confidence():
    Detection(scene_id="s", row=10, col=20, lon=1.0, lat=2.0, detection_confidence=0.9)
    with pytest.raises(ValidationError):
        Detection(scene_id="s", row=10, col=20, lon=1.0, lat=2.0, detection_confidence=1.5)


def test_ais_record_holds_kinematics():
    rec = AisRecord(
        mmsi=123456789, timestamp=dt.datetime(2026, 1, 1, tzinfo=dt.UTC), lon=1.0, lat=2.0
    )
    assert rec.mmsi == 123456789


def test_matched_contact_label_constrained():
    det = Detection(scene_id="s", row=1, col=1, lon=0.0, lat=0.0, detection_confidence=0.5)
    mc = MatchedContact(
        detection=det, label="dark", match_score=0.1, mmsi=None, dist_m=None, dt_s=None
    )
    assert mc.label == "dark"
    with pytest.raises(ValidationError):
        MatchedContact(
            detection=det,
            label="bogus",  # ty: ignore[invalid-argument-type]
            match_score=0.1,
            mmsi=None,
            dist_m=None,
            dt_s=None,
        )
