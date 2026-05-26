"""Threshold detector standing in for the real model in the walking skeleton."""

from __future__ import annotations

import numpy as np

from septentrion.io.synthetic import _pixel_to_lonlat
from septentrion.schemas import Detection


class StubDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def detect(self, raster: np.ndarray, scene_id: str) -> list[Detection]:
        rows, cols = np.where(raster >= self.threshold)
        out: list[Detection] = []
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            lon, lat = _pixel_to_lonlat(r, c)
            conf = float(min(1.0, raster[r, c]))
            out.append(
                Detection(
                    scene_id=scene_id, row=r, col=c, lon=lon, lat=lat, detection_confidence=conf
                )
            )
        return out
