"""Integration test: the packaged detector reproduces the verified real-scene result.

Skipped unless PyTorch and the credentialed assets are present (run
``uv sync --extra cpu`` and ``scripts/setup_xview3.sh``, and place the validation
scene ``590dd08f71056cacv`` + ``validation.csv`` under ``data/xview3/``). Marked
``integration`` and ``slow`` so it is excluded from the default fast suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # detector needs the optional torch extra

from septentrion.detect.xview3 import (
    DEFAULT_AI2_SRC,
    DEFAULT_WEIGHTS,
    Xview3Detector,
)
from septentrion.io.xview3_scene import XView3Scene

_REPO = Path(__file__).resolve().parents[2]
_SCENE_ID = "590dd08f71056cacv"
_SCENE_DIR = _REPO / "data" / "xview3" / "scenes" / _SCENE_ID
_LABELS = _REPO / "data" / "xview3" / "validation.csv"
# Window (verified) centred on a dense HIGH-confidence vessel cluster.
_ROW_OFF, _COL_OFF, _WIN = 7599, 4047, 3072

_assets_present = (
    DEFAULT_AI2_SRC.exists()
    and DEFAULT_WEIGHTS.exists()
    and _SCENE_DIR.exists()
    and _LABELS.exists()
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _assets_present, reason="xView3 assets not provisioned"),
]


def test_detector_recall_and_localization_on_high_confidence_window():
    import pandas as pd

    scene = XView3Scene(_SCENE_DIR)
    detector = Xview3Detector(device="cpu", score_threshold=0.2)

    # Run the packaged loader + detector on the one verified window.
    stacked = scene.read_window(_ROW_OFF, _COL_OFF, _WIN)
    centres, scores = detector._infer_window(stacked)
    keep = scores >= detector.score_threshold
    det_rows = centres[keep, 0] + _ROW_OFF
    det_cols = centres[keep, 1] + _COL_OFF
    assert keep.sum() > 0, "detector produced no detections on a vessel-dense window"

    df = pd.read_csv(_LABELS)
    g = df[df.scene_id.astype(str) == _SCENE_ID]
    in_win = g[
        (g.detect_scene_row >= _ROW_OFF)
        & (g.detect_scene_row < _ROW_OFF + _WIN)
        & (g.detect_scene_column >= _COL_OFF)
        & (g.detect_scene_column < _COL_OFF + _WIN)
    ]
    high = in_win[(in_win.confidence == "HIGH") & (in_win.is_vessel)]
    assert len(high) >= 20, "expected a dense HIGH-confidence cluster in this window"

    gr = high.detect_scene_row.to_numpy()
    gc = high.detect_scene_column.to_numpy()
    nearest = [
        float(np.min(np.hypot(det_cols - c, det_rows - r))) for r, c in zip(gr, gc, strict=True)
    ]
    matched = sum(d <= 20 for d in nearest)

    recall = matched / len(high)
    median_loc = float(np.median(nearest))
    assert recall >= 0.9, f"HIGH-confidence recall {recall:.2f} < 0.9"
    assert median_loc <= 3.0, f"median localization {median_loc:.1f}px > 3px"
