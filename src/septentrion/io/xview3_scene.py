"""Windowed reader for an xView3 analysis-ready scene.

An xView3 scene is a directory of co-registered GeoTIFFs: ``VV_dB.tif`` and
``VH_dB.tif`` (full-resolution UTM SAR backscatter in dB) plus coarse ancillary
layers (``bathymetry.tif`` and OWI wind fields). The AI2 xView3 detector consumes
the three channels ``[vh, vv, bathymetry]`` normalised to ``[0, 1]`` via the
``CustomNormalize2`` scheme. This loader reads arbitrary windows without holding a
full ~29k x 24k scene in memory, and maps pixel coordinates to WGS84 lon/lat.

Deliberately torch-free (uses GDAL/rasterio resampling) so it imports and tests
without the optional PyTorch dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

# rasterio's attrs-based Window is not typed for static checkers; alias as Any so
# keyword construction below isn't flagged.
_Window: Any = Window

# Channel order expected by the detector, and the file each channel comes from.
CHANNELS: tuple[str, ...] = ("vh", "vv", "bathymetry")
_FILENAMES = {"vh": "VH_dB.tif", "vv": "VV_dB.tif", "bathymetry": "bathymetry.tif"}


def normalize_channels(stacked: np.ndarray) -> np.ndarray:
    """Apply xView3 ``CustomNormalize2`` in place to a ``[vh, vv, bathymetry]`` stack.

    SAR channels are clipped to [-50, 20] dB then mapped to [0, 1]; bathymetry is
    clipped to [-6000, 2000] m then mapped to [0, 1].
    """
    stacked[0] = (np.clip(stacked[0], -50.0, 20.0) + 50.0) / 70.0
    stacked[1] = (np.clip(stacked[1], -50.0, 20.0) + 50.0) / 70.0
    stacked[2] = (np.clip(stacked[2], -6000.0, 2000.0) + 6000.0) / 8000.0
    return stacked


class XView3Scene:
    """A single xView3 scene directory, read lazily by window.

    Args:
        scene_dir: directory containing ``VV_dB.tif``, ``VH_dB.tif``, ``bathymetry.tif``.

    The VH raster defines the scene grid (``width`` x ``height``, CRS, transform);
    the coarse bathymetry raster is resampled to each requested window.
    """

    def __init__(self, scene_dir: str | Path) -> None:
        self.dir = Path(scene_dir)
        for name in _FILENAMES.values():
            if not (self.dir / name).exists():
                raise FileNotFoundError(f"missing channel {name} in {self.dir}")
        with rasterio.open(self.dir / _FILENAMES["vh"]) as ds:
            self.width = ds.width
            self.height = ds.height
            self._transform = ds.transform
            self._crs = ds.crs
        self._to_wgs84 = Transformer.from_crs(self._crs, "EPSG:4326", always_xy=True)

    def _read_sar(self, channel: str, row_off: int, col_off: int, size: int) -> np.ndarray:
        with rasterio.open(self.dir / _FILENAMES[channel]) as ds:
            win = _Window(col_off=col_off, row_off=row_off, width=size, height=size)
            return ds.read(1, window=win).astype("float32")

    def _read_bathymetry(self, row_off: int, col_off: int, size: int) -> np.ndarray:
        # The coarse bathymetry raster spans the same extent as the SAR grid; read the
        # matching (fractional) window and let GDAL resample it up to the window size.
        with rasterio.open(self.dir / _FILENAMES["bathymetry"]) as ds:
            bw, bh = ds.width, ds.height
            win = _Window(
                col_off=col_off * bw / self.width,
                row_off=row_off * bh / self.height,
                width=size * bw / self.width,
                height=size * bh / self.height,
            )
            return ds.read(
                1, window=win, out_shape=(size, size), resampling=Resampling.bilinear
            ).astype("float32")

    def read_window(self, row_off: int, col_off: int, size: int) -> np.ndarray:
        """Return a normalised ``(3, size, size)`` float32 stack ``[vh, vv, bathymetry]``."""
        vh = self._read_sar("vh", row_off, col_off, size)
        vv = self._read_sar("vv", row_off, col_off, size)
        bath = self._read_bathymetry(row_off, col_off, size)
        stacked = np.stack([vh, vv, bath], axis=0)
        return normalize_channels(stacked)

    def pixel_to_lonlat(self, row: float, col: float) -> tuple[float, float]:
        """Map a scene pixel (row, col) to (lon, lat) in WGS84 degrees."""
        x, y = self._transform * (col + 0.5, row + 0.5)
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)
