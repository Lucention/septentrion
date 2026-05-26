"""Detector interface — the real xView3 detector slots in here later (E1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from septentrion.schemas import Detection


class Detector(Protocol):
    def detect(self, raster: np.ndarray, scene_id: str) -> list[Detection]: ...
