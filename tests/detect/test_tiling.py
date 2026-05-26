"""Hermetic tests for window tiling and proximity dedup (no torch needed)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from septentrion.detect.tiling import dedup_by_proximity, window_offsets


def test_single_window_when_extent_fits():
    assert window_offsets(3072, 3072, 400) == [0]
    assert window_offsets(1000, 3072, 400) == [0]


def test_offsets_cover_extent_with_overlap():
    offs = window_offsets(10000, 3072, 400)
    assert offs[0] == 0
    assert offs[-1] == 10000 - 3072  # flush with far edge
    # consecutive windows overlap (step < window_size)
    assert all(b - a == 3072 - 800 for a, b in pairwise(offs) if b != offs[-1])
    # every column is covered by some window
    covered = [False] * 10000
    for o in offs:
        for i in range(o, o + 3072):
            covered[i] = True
    assert all(covered)


def test_window_offsets_rejects_bad_padding():
    with pytest.raises(ValueError):
        window_offsets(10000, 800, 400)  # step == 0


def test_dedup_keeps_highest_and_drops_near():
    dets = [(100.0, 100.0, 0.9), (104.0, 100.0, 0.5), (500.0, 500.0, 0.7)]
    kept = dedup_by_proximity(dets, dedup_px=10.0)
    assert (100.0, 100.0, 0.9) in kept
    assert (500.0, 500.0, 0.7) in kept
    assert (104.0, 100.0, 0.5) not in kept  # within 10px of the higher-scoring one
    assert len(kept) == 2


def test_dedup_keeps_distant_duplicates():
    dets = [(0.0, 0.0, 0.8), (50.0, 0.0, 0.8)]
    assert len(dedup_by_proximity(dets, dedup_px=10.0)) == 2
