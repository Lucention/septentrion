import datetime as dt

from septentrion.correlate.spatial import correlate
from septentrion.schemas import AisRecord, Detection


def _det(lon, lat):
    return Detection(scene_id="s", row=0, col=0, lon=lon, lat=lat, detection_confidence=1.0)


def test_close_ais_matches_far_is_dark():
    ts = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    dets = [_det(10.0, 60.0), _det(20.0, 60.0)]
    ais = [AisRecord(mmsi=1, timestamp=ts, lon=10.0001, lat=60.0001)]
    contacts = correlate(dets, ais, max_dist_m=2000.0)
    by_label = {c.detection.lon: c.label for c in contacts}
    assert by_label[10.0] == "matched"
    assert by_label[20.0] == "dark"
    matched = next(c for c in contacts if c.label == "matched")
    assert matched.mmsi == 1
    assert matched.dist_m is not None and matched.dist_m < 2000.0
