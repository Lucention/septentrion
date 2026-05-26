import numpy as np

from septentrion.config import SyntheticCfg
from septentrion.io.synthetic import SyntheticScene, generate_scene


def test_generate_scene_shapes_and_counts():
    rng = np.random.default_rng(0)
    cfg = SyntheticCfg(n_vessels=20, dark_fraction=0.25, raster_size=128)
    scene: SyntheticScene = generate_scene(rng, cfg, scene_id="syn-0")

    assert scene.raster.shape == (128, 128)
    assert len(scene.truth) == 20
    n_dark = sum(1 for d in scene.truth if d.__dict__.get("label") == "dark")
    assert n_dark == 5  # 0.25 * 20
    # only matched vessels broadcast AIS
    assert len(scene.ais) == 15


def test_bright_pixels_present_at_vessels():
    rng = np.random.default_rng(1)
    cfg = SyntheticCfg(n_vessels=10, dark_fraction=0.0, raster_size=64)
    scene = generate_scene(rng, cfg, scene_id="syn-1")
    for d in scene.truth:
        assert scene.raster[d.row, d.col] > 0.5
