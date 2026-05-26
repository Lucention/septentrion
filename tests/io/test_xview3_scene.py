"""Hermetic tests for the xView3 scene loader using tiny synthetic GeoTIFFs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from septentrion.io.xview3_scene import XView3Scene, normalize_channels

_UTM31N = "EPSG:32631"


def _write_tif(path: Path, arr: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=_UTM31N,
        transform=transform,
    ) as ds:
        ds.write(arr.astype("float32"), 1)


@pytest.fixture
def synthetic_scene(tmp_path: Path) -> Path:
    # Full-res SAR grid 64x64 at 10 m, origin in UTM zone 31N.
    transform = from_origin(500_000, 5_600_000, 10, 10)
    vh = np.full((64, 64), -15.0, dtype="float32")  # -> 0.5 after CustomNormalize2
    vh[0, 0] = -50.0  # -> 0.0
    vh[0, 1] = 20.0  # -> 1.0
    vv = np.full((64, 64), 5.0, dtype="float32")
    _write_tif(tmp_path / "VH_dB.tif", vh, transform)
    _write_tif(tmp_path / "VV_dB.tif", vv, transform)
    # Coarse bathymetry 8x8 over the same extent.
    bath_transform = from_origin(500_000, 5_600_000, 80, 80)
    _write_tif(
        tmp_path / "bathymetry.tif", np.full((8, 8), -2000.0, dtype="float32"), bath_transform
    )
    return tmp_path


def test_normalize_channels_maps_to_unit_interval():
    stacked = np.array([[[-50.0, 20.0]], [[-50.0, 20.0]], [[-6000.0, 2000.0]]], dtype="float32")
    out = normalize_channels(stacked.copy())
    np.testing.assert_allclose(out[0], [[0.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(out[1], [[0.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(out[2], [[0.0, 1.0]], atol=1e-6)


def test_scene_dims_and_missing_channel(tmp_path: Path, synthetic_scene: Path):
    scene = XView3Scene(synthetic_scene)
    assert (scene.width, scene.height) == (64, 64)
    with pytest.raises(FileNotFoundError):
        XView3Scene(tmp_path / "does_not_exist")


def test_read_window_shape_and_normalization(synthetic_scene: Path):
    scene = XView3Scene(synthetic_scene)
    win = scene.read_window(0, 0, 64)
    assert win.shape == (3, 64, 64)
    assert win.dtype == np.float32
    assert win.min() >= 0.0 and win.max() <= 1.0
    # VH normalization landmarks
    assert win[0, 0, 0] == pytest.approx(0.0, abs=1e-4)  # -50 dB
    assert win[0, 0, 1] == pytest.approx(1.0, abs=1e-4)  # 20 dB
    assert win[0, 1, 1] == pytest.approx(0.5, abs=1e-4)  # -15 dB elsewhere


def test_pixel_to_lonlat_is_plausible_and_monotonic(synthetic_scene: Path):
    scene = XView3Scene(synthetic_scene)
    lon0, lat0 = scene.pixel_to_lonlat(0, 0)
    # UTM 31N origin (500000, 5600000) sits near 3degE, ~50.5degN.
    assert 1.0 < lon0 < 5.0
    assert 49.0 < lat0 < 52.0
    lon_e, _ = scene.pixel_to_lonlat(0, 40)  # move east in columns -> larger lon
    _, lat_s = scene.pixel_to_lonlat(40, 0)  # move south in rows -> smaller lat
    assert lon_e > lon0
    assert lat_s < lat0
