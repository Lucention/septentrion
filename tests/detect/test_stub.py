import numpy as np

from septentrion.detect.stub import StubDetector


def test_stub_detects_bright_pixels():
    raster = np.zeros((16, 16), dtype="float32")
    raster[5, 7] = 1.0
    raster[10, 2] = 0.9
    det = StubDetector(threshold=0.5)
    out = det.detect(raster, scene_id="s")
    coords = {(d.row, d.col) for d in out}
    assert (5, 7) in coords
    assert (10, 2) in coords
    assert all(0.0 <= d.detection_confidence <= 1.0 for d in out)
