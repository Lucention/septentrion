"""Write artifacts with a traceable `<name>.prov.json` sidecar."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from septentrion.config import Config

_TRACKED = ("numpy", "scikit-learn", "rasterio", "torch")


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:  # provenance must never crash a run
        return "unknown"


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _TRACKED:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def write_artifact(path: str | Path, payload: bytes, *, cfg: Config, inputs: list[str]) -> Path:
    """Write `payload` to `path` and a provenance sidecar next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    meta = {
        "artifact": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": _git_commit(),
        "input_scene_ids": inputs,
        "config": cfg.model_dump(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": _versions(),
    }
    sidecar = path.with_suffix(path.suffix + ".prov.json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
