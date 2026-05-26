import datetime as dt

from septentrion.correlate.spatial import correlate
from septentrion.schemas import AisRecord, Detection
from septentrion.score.darkness import score_contacts, separation_auc


def _det(lon, lat):
    return Detection(scene_id="s", row=0, col=0, lon=lon, lat=lat, detection_confidence=1.0)


def test_scores_in_unit_interval_and_separate():
    ts = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    dets = [_det(10.0, 60.0), _det(20.0, 60.0)]
    ais = [AisRecord(mmsi=1, timestamp=ts, lon=10.0, lat=60.0)]
    contacts = correlate(dets, ais, max_dist_m=2000.0)
    scored = score_contacts(contacts, max_dist_m=2000.0)
    assert all(0.0 <= c.match_score <= 1.0 for c in scored)
    truth = [c.label == "matched" for c in scored]
    auc = separation_auc(scored, truth)
    assert auc == 1.0
