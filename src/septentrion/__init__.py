"""Septentrion Aperture — Arctic MDA cross-modal fusion proof-of-concept."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("septentrion")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
