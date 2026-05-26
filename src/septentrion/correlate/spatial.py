"""Minimal nearest-AIS spatial matcher (E2 replaces this with spatiotemporal logic)."""

from __future__ import annotations

from septentrion.geo.distance import haversine_m
from septentrion.schemas import AisRecord, Detection, MatchedContact


def correlate(
    detections: list[Detection], ais: list[AisRecord], max_dist_m: float
) -> list[MatchedContact]:
    """Label each detection matched (nearest AIS within `max_dist_m`) or dark."""
    contacts: list[MatchedContact] = []
    for det in detections:
        best_mmsi: int | None = None
        best_dist = float("inf")
        for rec in ais:
            d = haversine_m(det.lon, det.lat, rec.lon, rec.lat)
            if d < best_dist:
                best_dist, best_mmsi = d, rec.mmsi
        if best_mmsi is not None and best_dist <= max_dist_m:
            contacts.append(
                MatchedContact(
                    detection=det,
                    label="matched",
                    match_score=0.0,
                    mmsi=best_mmsi,
                    dist_m=best_dist,
                    dt_s=None,
                )
            )
        else:
            contacts.append(
                MatchedContact(detection=det, label="dark", match_score=0.0, mmsi=None, dist_m=None)
            )
    return contacts
