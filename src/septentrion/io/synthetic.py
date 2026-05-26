"""Synthetic Sentinel-1-like scene for the walking skeleton (no real data needed)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from septentrion.schemas import AisRecord, Detection

if TYPE_CHECKING:
    from septentrion.config import SyntheticCfg

# Each pixel maps to a fixed lon/lat step so positions are deterministic.
_LON0, _LAT0, _STEP_DEG = 10.0, 60.0, 0.002


def _pixel_to_lonlat(row: int, col: int) -> tuple[float, float]:
    return _LON0 + col * _STEP_DEG, _LAT0 + row * _STEP_DEG


@dataclass
class SyntheticScene:
    scene_id: str
    timestamp: dt.datetime
    raster: np.ndarray
    ais: list[AisRecord]
    truth: list[Detection]  # detection_confidence holds the true brightness


def generate_scene(rng: np.random.Generator, cfg: SyntheticCfg, scene_id: str) -> SyntheticScene:
    n = cfg.n_vessels
    size = cfg.raster_size
    raster = rng.normal(0.1, 0.02, size=(size, size)).astype("float32")

    rows = rng.integers(1, size - 1, size=n)
    cols = rng.integers(1, size - 1, size=n)
    n_dark = round(cfg.dark_fraction * n)
    is_dark = np.zeros(n, dtype=bool)
    is_dark[:n_dark] = True
    rng.shuffle(is_dark)

    ts = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
    truth: list[Detection] = []
    ais: list[AisRecord] = []
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        raster[r, c] = 1.0  # bright vessel return
        lon, lat = _pixel_to_lonlat(r, c)
        label = "dark" if is_dark[i] else "matched"
        truth.append(
            Detection(scene_id=scene_id, row=r, col=c, lon=lon, lat=lat, detection_confidence=1.0)
        )
        truth[-1].__dict__["label"] = label  # carry truth label for evaluation
        if label == "matched":
            ais.append(AisRecord(mmsi=200_000_000 + i, timestamp=ts, lon=lon, lat=lat))

    return SyntheticScene(scene_id=scene_id, timestamp=ts, raster=raster, ais=ais, truth=truth)
