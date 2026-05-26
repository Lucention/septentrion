"""Pure-numpy helpers for tiling a scene into windows and deduplicating detections.

Kept torch-free so the windowing/NMS logic can be unit-tested without the optional
PyTorch dependency that the detector itself needs.
"""

from __future__ import annotations

import numpy as np


def window_offsets(extent: int, window_size: int, padding: int) -> list[int]:
    """Start offsets tiling ``extent`` with ``window_size`` windows that overlap by ``2*padding``.

    Always covers the full extent: the final offset is flush with the far edge. If the
    extent fits in one window, returns ``[0]``.
    """
    if extent <= window_size:
        return [0]
    step = window_size - 2 * padding
    if step <= 0:
        raise ValueError("window_size must exceed 2*padding")
    offsets = list(range(0, extent - window_size, step))
    offsets.append(extent - window_size)
    return offsets


def dedup_by_proximity(
    detections: list[tuple[float, float, float]], dedup_px: float
) -> list[tuple[float, float, float]]:
    """Greedy NMS by proximity over ``(row, col, score)`` tuples.

    Keeps the highest-scoring detection and drops any later one within ``dedup_px``.
    """
    kept: list[tuple[float, float, float]] = []
    for row, col, score in sorted(detections, key=lambda t: t[2], reverse=True):
        if all(np.hypot(row - kr, col - kc) > dedup_px for kr, kc, _ in kept):
            kept.append((row, col, score))
    return kept
