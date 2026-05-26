# Foundation & Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `septentrion` installable package with full Python tooling, then build a synthetic walking skeleton that flows fake data through detect → correlate → score and writes a provenance-tracked result — runnable and fully tested on Windows/Linux/macOS with no GPU and no real data.

**Architecture:** `src/`-layout package managed by `uv`. Foundation modules (config, rng, provenance, logging) underpin typed data contracts (`schemas.py`). A synthetic generator produces a tiny raster + AIS + ground-truth labels; a stub detector emits detections; a minimal spatial correlate labels each matched/dark; a minimal score plus separation metric run on CPU. A `typer` CLI exposes `run-skeleton`. Real loaders, CRS handling, spatiotemporal correlation, the real detector, E4, and reporting are later plans that swap in behind these interfaces.

**Tech Stack:** Python 3.14, uv, hatchling, ruff, ty, pytest, pydantic-settings, structlog, numpy, scikit-learn, typer. (torch/rasterio/geopandas are declared but NOT exercised by the skeleton.)

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, optional torch extras, dependency groups, tool configs (ruff/ty/pytest/coverage) |
| `.python-version` | Pin interpreter to 3.14 |
| `.gitignore` | Ignore `data/`, `outputs/`, `.venv/`, `*.ipynb`, caches |
| `.pre-commit-config.yaml` | ruff, hygiene, large-file block, nbstripout |
| `.github/workflows/ci.yml` | 3-OS matrix: ruff, ty, fast tests |
| `README.md` | Quickstart |
| `src/septentrion/__init__.py` | Light package init, `__version__` |
| `src/septentrion/config.py` | `pydantic-settings` `Config` with nested per-stage models |
| `src/septentrion/rng.py` | `set_seed()` → `np.random.Generator` |
| `src/septentrion/provenance.py` | `write_artifact()` + `<name>.prov.json` sidecar |
| `src/septentrion/logging_setup.py` | structlog configuration |
| `src/septentrion/schemas.py` | `Detection`, `AisRecord`, `MatchedContact` typed models |
| `src/septentrion/geo/distance.py` | Haversine distance in metres |
| `src/septentrion/io/synthetic.py` | Synthetic scene generator (raster + AIS + truth) |
| `src/septentrion/detect/base.py` | `Detector` protocol |
| `src/septentrion/detect/stub.py` | `StubDetector` (bright-pixel detector) |
| `src/septentrion/correlate/spatial.py` | Minimal nearest-AIS spatial matcher |
| `src/septentrion/score/darkness.py` | Distance-based score + separation metric |
| `src/septentrion/pipelines/skeleton.py` | End-to-end synthetic run |
| `src/septentrion/cli.py` | `typer` app, `run-skeleton` command |
| `tests/...` | Mirror of `src/septentrion/`, plus `conftest.py` |

---

## Task 1: Project scaffold (package installs)

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/septentrion/__init__.py`

- [ ] **Step 1: Ensure `uv` is installed**

Run (macOS/Linux): `curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version`
Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
Expected: prints a uv version (≥ 0.5.3).

- [ ] **Step 2: Create `.python-version`**

```
3.14
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "septentrion"
version = "0.1.0"
description = "Septentrion Aperture — Arctic MDA cross-modal fusion proof-of-concept"
readme = "README.md"
requires-python = ">=3.14"
license = "MIT"
authors = [{ name = "Mike Ross", email = "mike@getboosted.io" }]
dependencies = [
    "numpy>=2.1,<3",
    "pandas>=2.2,<4",
    "scikit-learn>=1.6",
    "matplotlib>=3.9",
    "networkx>=3.4",
    "rasterio>=1.5,<1.6",
    "shapely>=2.1,<3",
    "pyproj>=3.7,<4",
    "geopandas>=1.1,<2",
    "pyogrio>=0.11",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "structlog>=24.4",
    "typer>=0.15",
]

[project.optional-dependencies]
cpu = ["torch>=2.6", "torchvision>=0.21"]
cu128 = ["torch>=2.6", "torchvision>=0.21"]

[project.scripts]
septentrion = "septentrion.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-cov>=6.0",
    "pytest-xdist>=3.6",
    "ruff>=0.9",
    "ty>=0.0.1a1",
    "pre-commit>=4.0",
    "ipykernel>=6.29",
    "jupytext>=1.16",
]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu128" }]]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu128", extra = "cu128", marker = "sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu128", extra = "cu128", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.hatch.build.targets.wheel]
packages = ["src/septentrion"]

[tool.ruff]
src = ["src", "tests"]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "C4", "SIM", "RET", "PTH", "NPY", "PD", "RUF", "TID", "TC"]
ignore = ["E501", "B008", "SIM108", "RET504"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["B011", "PD901"]
"**/*.ipynb" = ["E402", "F401", "F811", "B018"]
"__init__.py" = ["F401"]

[tool.ruff.lint.isort]
known-first-party = ["septentrion"]

[tool.ruff.format]
docstring-code-format = true

[tool.ty.environment]
root = ["./src"]

[tool.ty.rules]
unresolved-import = "ignore"

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
pythonpath = ["src"]
addopts = ["-ra", "--strict-markers", "--strict-config", "-n", "auto", "-m", "not slow and not gpu"]
markers = [
    "slow: long-running tests",
    "gpu: requires a CUDA device",
    "integration: needs heavy optional deps (gdal/torch)",
]

[tool.coverage.run]
source = ["septentrion"]
branch = true
parallel = true

[tool.coverage.report]
show_missing = true
skip_covered = true
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
# Environments
.venv/
__pycache__/
*.py[cod]

# Tooling caches
.pytest_cache/
.ruff_cache/
.ty_cache/
.coverage
coverage.xml

# Project data & outputs (never commit)
data/
outputs/

# Notebooks: keep paired .py, ignore the .ipynb
*.ipynb
```

- [ ] **Step 5: Create `README.md`**

```markdown
# Septentrion Aperture

TRL-3 proof-of-concept for IDEaS CFP6 Ch13 — Arctic Maritime Domain Awareness.
Cross-modal SAR + AIS fusion: detection (E1), SAR–AIS correlation (E2),
confidence/darkness score (E3), entity-resolution graph stub (E4).

## Quickstart

```bash
uv sync                 # base deps (no torch) — runs the fusion layer
uv run septentrion run-skeleton   # synthetic end-to-end smoke run
uv run pytest           # fast test suite
```

Add the detector's PyTorch dependency only when needed:
`uv sync --extra cpu` (any platform) or `uv sync --extra cu128` (Linux GPU).
```

- [ ] **Step 6: Create `src/septentrion/__init__.py`**

```python
"""Septentrion Aperture — Arctic MDA cross-modal fusion proof-of-concept."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("septentrion")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
```

- [ ] **Step 7: Sync and verify the package imports**

Run: `uv sync && uv run python -c "import septentrion; print(septentrion.__version__)"`
Expected: prints `0.1.0` (creates `.venv`, `uv.lock`).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .python-version .gitignore README.md src/septentrion/__init__.py uv.lock
git commit -m "chore: scaffold septentrion package with uv + tooling config"
```

---

## Task 2: Tooling — pre-commit and CI

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ["--maxkb=1024"]
      - id: debug-statements
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.1
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout
    rev: 0.8.1
    hooks:
      - id: nbstripout
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
    branches: [master]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - name: Install project + dev deps
        run: uv sync
      - name: Ruff lint
        run: uv run ruff check --output-format=github .
      - name: Ruff format check
        run: uv run ruff format --check .
      - name: Type check (ty)
        run: uv run ty check
      - name: Tests (fast, CPU-only)
        run: uv run pytest --cov --cov-report=xml   # -n auto comes from addopts
```

- [ ] **Step 3: Verify ruff and ty run clean on the current tree**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: ruff reports no errors; ty reports success (no source files yet beyond `__init__.py`).

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml .github/workflows/ci.yml
git commit -m "ci: add pre-commit hooks and 3-OS GitHub Actions matrix"
```

---

## Task 3: Configuration module

**Files:**
- Create: `src/septentrion/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from septentrion.config import Config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.seed == 42
    assert 0.0 <= cfg.scoring.threshold <= 1.0
    assert cfg.correlation.max_dist_m > 0


def test_loads_from_toml(tmp_path: Path):
    toml = tmp_path / "config.toml"
    toml.write_text("seed = 7\n[scoring]\nthreshold = 0.9\n", encoding="utf-8")
    cfg = Config.from_toml(toml)
    assert cfg.seed == 7
    assert cfg.scoring.threshold == 0.9


def test_model_dump_is_serializable():
    import json

    json.dumps(Config().model_dump())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/config.py
"""Validated, provenance-friendly configuration for the PoC pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestCfg(BaseModel):
    raw_dir: str = "data/raw"
    scene_ids: list[str] = Field(default_factory=list)


class DetectionCfg(BaseModel):
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class CorrelationCfg(BaseModel):
    max_dist_m: float = Field(2000.0, gt=0)
    max_dt_s: float = Field(1800.0, gt=0)


class ScoringCfg(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class SyntheticCfg(BaseModel):
    n_vessels: int = Field(40, gt=0)
    dark_fraction: float = Field(0.3, ge=0.0, le=1.0)
    raster_size: int = Field(256, gt=0)


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEPT_", env_nested_delimiter="__")

    seed: int = 42
    out_dir: str = "outputs"
    ingest: IngestCfg = IngestCfg()
    detection: DetectionCfg = DetectionCfg()
    correlation: CorrelationCfg = CorrelationCfg()
    scoring: ScoringCfg = ScoringCfg()
    synthetic: SyntheticCfg = SyntheticCfg()

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        import tomllib

        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/config.py tests/test_config.py
git commit -m "feat: add validated pydantic-settings configuration"
```

---

## Task 4: Seed control

**Files:**
- Create: `src/septentrion/rng.py`
- Test: `tests/test_rng.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rng.py
import numpy as np

from septentrion.rng import set_seed


def test_set_seed_returns_generator():
    rng = set_seed(123)
    assert isinstance(rng, np.random.Generator)


def test_set_seed_is_reproducible():
    a = set_seed(123).random(5)
    b = set_seed(123).random(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = set_seed(1).random(5)
    b = set_seed(2).random(5)
    assert not np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rng.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.rng'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/rng.py
"""Deterministic seeding across stdlib, NumPy, and (optionally) PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed all RNGs and return a NumPy Generator to thread through stages."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return np.random.default_rng(seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rng.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/rng.py tests/test_rng.py
git commit -m "feat: add deterministic set_seed helper"
```

---

## Task 5: Provenance sidecar

**Files:**
- Create: `src/septentrion/provenance.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provenance.py
import hashlib
import json
from pathlib import Path

from septentrion.config import Config
from septentrion.provenance import write_artifact


def test_write_artifact_writes_payload_and_sidecar(tmp_path: Path):
    out = tmp_path / "result.csv"
    payload = b"a,b\n1,2\n"
    write_artifact(out, payload, cfg=Config(), inputs=["scene_A"])

    assert out.read_bytes() == payload
    sidecar = out.with_suffix(out.suffix + ".prov.json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))

    assert meta["artifact"] == "result.csv"
    assert meta["sha256"] == hashlib.sha256(payload).hexdigest()
    assert meta["input_scene_ids"] == ["scene_A"]
    assert meta["config"]["seed"] == 42
    assert "git_commit" in meta
    assert "created_utc" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.provenance'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/provenance.py
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
    except Exception:  # noqa: BLE001 - provenance must never crash a run
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/provenance.py tests/test_provenance.py
git commit -m "feat: add provenance sidecar writer"
```

---

## Task 6: Logging setup

**Files:**
- Create: `src/septentrion/logging_setup.py`
- Test: `tests/test_logging_setup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging_setup.py
import structlog

from septentrion.logging_setup import setup_logging


def test_setup_logging_binds_context_and_returns_logger():
    log = setup_logging(run_id="run-1", git_commit="abc123")
    assert log is not None
    # bound context vars are present on new loggers after setup
    ctx = structlog.contextvars.get_contextvars()
    assert ctx.get("run_id") == "run-1"
    assert ctx.get("git_commit") == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging_setup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.logging_setup'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/logging_setup.py
"""structlog configuration: human console by default, context bound per run."""

from __future__ import annotations

import logging

import structlog


def setup_logging(run_id: str, git_commit: str, level: int = logging.INFO) -> structlog.stdlib.BoundLogger:
    """Configure structlog and bind run-wide provenance context."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=run_id, git_commit=git_commit)
    return structlog.get_logger()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging_setup.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/logging_setup.py tests/test_logging_setup.py
git commit -m "feat: add structlog logging setup"
```

---

## Task 7: Data contracts

**Files:**
- Create: `src/septentrion/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
import datetime as dt

import pytest
from pydantic import ValidationError

from septentrion.schemas import AisRecord, Detection, MatchedContact


def test_detection_requires_valid_confidence():
    Detection(scene_id="s", row=10, col=20, lon=1.0, lat=2.0, detection_confidence=0.9)
    with pytest.raises(ValidationError):
        Detection(scene_id="s", row=10, col=20, lon=1.0, lat=2.0, detection_confidence=1.5)


def test_ais_record_holds_kinematics():
    rec = AisRecord(mmsi=123456789, timestamp=dt.datetime(2026, 1, 1, tzinfo=dt.UTC), lon=1.0, lat=2.0)
    assert rec.mmsi == 123456789


def test_matched_contact_label_constrained():
    det = Detection(scene_id="s", row=1, col=1, lon=0.0, lat=0.0, detection_confidence=0.5)
    mc = MatchedContact(detection=det, label="dark", match_score=0.1, mmsi=None, dist_m=None, dt_s=None)
    assert mc.label == "dark"
    with pytest.raises(ValidationError):
        MatchedContact(detection=det, label="bogus", match_score=0.1, mmsi=None, dist_m=None, dt_s=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.schemas'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/schemas.py
"""Typed data contracts passed between pipeline stages."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

MatchLabel = Literal["matched", "dark"]


class Detection(BaseModel):
    scene_id: str
    row: int
    col: int
    lon: float
    lat: float
    detection_confidence: float = Field(ge=0.0, le=1.0)


class AisRecord(BaseModel):
    mmsi: int
    timestamp: dt.datetime
    lon: float
    lat: float


class MatchedContact(BaseModel):
    detection: Detection
    label: MatchLabel
    match_score: float = Field(ge=0.0, le=1.0)
    mmsi: int | None = None
    dist_m: float | None = None
    dt_s: float | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/schemas.py tests/test_schemas.py
git commit -m "feat: add typed data contracts (Detection, AisRecord, MatchedContact)"
```

---

## Task 8: Haversine distance

**Files:**
- Create: `src/septentrion/geo/__init__.py`
- Create: `src/septentrion/geo/distance.py`
- Test: `tests/geo/test_distance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/geo/test_distance.py
from septentrion.geo.distance import haversine_m


def test_zero_distance():
    assert haversine_m(0.0, 0.0, 0.0, 0.0) == 0.0


def test_one_degree_latitude_is_about_111km():
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert 110_000 < d < 112_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/geo/test_distance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.geo'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/geo/__init__.py
```

```python
# src/septentrion/geo/distance.py
"""Great-circle distance in metres (skeleton stand-in for full CRS reprojection)."""

from __future__ import annotations

import math

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two lon/lat points, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/geo/test_distance.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/geo tests/geo/test_distance.py
git commit -m "feat: add haversine distance helper"
```

---

## Task 9: Synthetic scene generator

**Files:**
- Create: `src/septentrion/io/__init__.py`
- Create: `src/septentrion/io/synthetic.py`
- Test: `tests/io/test_synthetic.py`

The generator returns a small raster with bright pixels at vessel locations, AIS records for the *matched* fraction only (dark vessels broadcast nothing), and ground-truth detections labelled matched/dark.

- [ ] **Step 1: Write the failing test**

```python
# tests/io/test_synthetic.py
import numpy as np

from septentrion.config import SyntheticCfg
from septentrion.io.synthetic import SyntheticScene, generate_scene


def test_generate_scene_shapes_and_counts():
    rng = np.random.default_rng(0)
    cfg = SyntheticCfg(n_vessels=20, dark_fraction=0.25, raster_size=128)
    scene: SyntheticScene = generate_scene(rng, cfg, scene_id="syn-0")

    assert scene.raster.shape == (128, 128)
    assert len(scene.truth) == 20
    n_dark = sum(1 for d in scene.truth if d.label == "dark")
    assert n_dark == 5  # 0.25 * 20
    # only matched vessels broadcast AIS
    assert len(scene.ais) == 15


def test_bright_pixels_present_at_vessels():
    rng = np.random.default_rng(1)
    cfg = SyntheticCfg(n_vessels=10, dark_fraction=0.0, raster_size=64)
    scene = generate_scene(rng, cfg, scene_id="syn-1")
    for d in scene.truth:
        assert scene.raster[d.row, d.col] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/io/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.io'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/io/__init__.py
```

```python
# src/septentrion/io/synthetic.py
"""Synthetic Sentinel-1-like scene for the walking skeleton (no real data needed)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from septentrion.config import SyntheticCfg
from septentrion.schemas import AisRecord, Detection

# Each pixel maps to a fixed lon/lat step so positions are deterministic.
_LON0, _LAT0, _STEP_DEG = 10.0, 60.0, 0.002


def _pixel_to_lonlat(row: int, col: int) -> tuple[float, float]:
    return _LON0 + col * _STEP_DEG, _LAT0 + row * _STEP_DEG


@dataclass
class SyntheticScene:
    scene_id: str
    timestamp: dt.datetime
    raster: np.ndarray
    ais: list[AisRecord]
    truth: list[Detection]  # detection_confidence holds the true brightness


def generate_scene(rng: np.random.Generator, cfg: SyntheticCfg, scene_id: str) -> SyntheticScene:
    n = cfg.n_vessels
    size = cfg.raster_size
    raster = rng.normal(0.1, 0.02, size=(size, size)).astype("float32")

    rows = rng.integers(1, size - 1, size=n)
    cols = rng.integers(1, size - 1, size=n)
    n_dark = round(cfg.dark_fraction * n)
    is_dark = np.zeros(n, dtype=bool)
    is_dark[:n_dark] = True
    rng.shuffle(is_dark)

    ts = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
    truth: list[Detection] = []
    ais: list[AisRecord] = []
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        raster[r, c] = 1.0  # bright vessel return
        lon, lat = _pixel_to_lonlat(r, c)
        label = "dark" if is_dark[i] else "matched"
        truth.append(
            Detection(scene_id=scene_id, row=r, col=c, lon=lon, lat=lat, detection_confidence=1.0)
        )
        truth[-1].__dict__["label"] = label  # carry truth label for evaluation
        if label == "matched":
            ais.append(AisRecord(mmsi=200_000_000 + i, timestamp=ts, lon=lon, lat=lat))

    return SyntheticScene(scene_id=scene_id, timestamp=ts, raster=raster, ais=ais, truth=truth)
```

> Note: `truth` items are `Detection`s; their matched/dark ground truth is stashed in `__dict__["label"]` for the evaluation step (Task 12). The pipeline itself never reads it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/io/test_synthetic.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/io tests/io/test_synthetic.py
git commit -m "feat: add synthetic scene generator for walking skeleton"
```

---

## Task 10: Detector protocol + stub

**Files:**
- Create: `src/septentrion/detect/__init__.py`
- Create: `src/septentrion/detect/base.py`
- Create: `src/septentrion/detect/stub.py`
- Test: `tests/detect/test_stub.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/detect/test_stub.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detect/test_stub.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.detect'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/detect/__init__.py
```

```python
# src/septentrion/detect/base.py
"""Detector interface — the real xView3 detector slots in here later (E1)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from septentrion.schemas import Detection


class Detector(Protocol):
    def detect(self, raster: np.ndarray, scene_id: str) -> list[Detection]: ...
```

```python
# src/septentrion/detect/stub.py
"""Threshold detector standing in for the real model in the walking skeleton."""

from __future__ import annotations

import numpy as np

from septentrion.io.synthetic import _pixel_to_lonlat
from septentrion.schemas import Detection


class StubDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def detect(self, raster: np.ndarray, scene_id: str) -> list[Detection]:
        rows, cols = np.where(raster >= self.threshold)
        out: list[Detection] = []
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            lon, lat = _pixel_to_lonlat(r, c)
            conf = float(min(1.0, raster[r, c]))
            out.append(
                Detection(scene_id=scene_id, row=r, col=c, lon=lon, lat=lat, detection_confidence=conf)
            )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/detect/test_stub.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/detect tests/detect/test_stub.py
git commit -m "feat: add detector protocol and threshold stub detector"
```

---

## Task 11: Spatial correlation + darkness score

**Files:**
- Create: `src/septentrion/correlate/__init__.py`
- Create: `src/septentrion/correlate/spatial.py`
- Create: `src/septentrion/score/__init__.py`
- Create: `src/septentrion/score/darkness.py`
- Test: `tests/correlate/test_spatial.py`
- Test: `tests/score/test_darkness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/correlate/test_spatial.py
import datetime as dt

from septentrion.correlate.spatial import correlate
from septentrion.schemas import AisRecord, Detection


def _det(lon, lat):
    return Detection(scene_id="s", row=0, col=0, lon=lon, lat=lat, detection_confidence=1.0)


def test_close_ais_matches_far_is_dark():
    ts = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    dets = [_det(10.0, 60.0), _det(20.0, 60.0)]
    ais = [AisRecord(mmsi=1, timestamp=ts, lon=10.0001, lat=60.0001)]
    contacts = correlate(dets, ais, max_dist_m=2000.0)
    by_label = {c.detection.lon: c.label for c in contacts}
    assert by_label[10.0] == "matched"
    assert by_label[20.0] == "dark"
    matched = next(c for c in contacts if c.label == "matched")
    assert matched.mmsi == 1
    assert matched.dist_m is not None and matched.dist_m < 2000.0
```

```python
# tests/score/test_darkness.py
import datetime as dt

from septentrion.correlate.spatial import correlate
from septentrion.schemas import AisRecord, Detection
from septentrion.score.darkness import score_contacts, separation_auc


def _det(lon, lat):
    return Detection(scene_id="s", row=0, col=0, lon=lon, lat=lat, detection_confidence=1.0)


def test_scores_in_unit_interval_and_separate():
    ts = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    dets = [_det(10.0, 60.0), _det(20.0, 60.0)]
    ais = [AisRecord(mmsi=1, timestamp=ts, lon=10.0, lat=60.0)]
    contacts = correlate(dets, ais, max_dist_m=2000.0)
    scored = score_contacts(contacts, max_dist_m=2000.0)
    assert all(0.0 <= c.match_score <= 1.0 for c in scored)
    truth = [c.label == "matched" for c in scored]
    auc = separation_auc(scored, truth)
    assert auc == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/correlate tests/score -v`
Expected: FAIL with `ModuleNotFoundError` for `septentrion.correlate` / `septentrion.score`.

- [ ] **Step 3: Write minimal implementations**

```python
# src/septentrion/correlate/__init__.py
```

```python
# src/septentrion/correlate/spatial.py
"""Minimal nearest-AIS spatial matcher (E2 replaces this with spatiotemporal logic)."""

from __future__ import annotations

from septentrion.geo.distance import haversine_m
from septentrion.schemas import AisRecord, Detection, MatchedContact


def correlate(
    detections: list[Detection], ais: list[AisRecord], max_dist_m: float
) -> list[MatchedContact]:
    """Label each detection matched (nearest AIS within `max_dist_m`) or dark."""
    contacts: list[MatchedContact] = []
    for det in detections:
        best_mmsi: int | None = None
        best_dist = float("inf")
        for rec in ais:
            d = haversine_m(det.lon, det.lat, rec.lon, rec.lat)
            if d < best_dist:
                best_dist, best_mmsi = d, rec.mmsi
        if best_mmsi is not None and best_dist <= max_dist_m:
            contacts.append(
                MatchedContact(
                    detection=det,
                    label="matched",
                    match_score=0.0,
                    mmsi=best_mmsi,
                    dist_m=best_dist,
                    dt_s=None,
                )
            )
        else:
            contacts.append(
                MatchedContact(detection=det, label="dark", match_score=0.0, mmsi=None, dist_m=None)
            )
    return contacts
```

```python
# src/septentrion/score/__init__.py
```

```python
# src/septentrion/score/darkness.py
"""Confidence/darkness score + separation metric (E3 extends with PR/ROC figures)."""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from septentrion.schemas import MatchedContact


def score_contacts(contacts: list[MatchedContact], max_dist_m: float) -> list[MatchedContact]:
    """Assign a match_score in [0,1]: 1 = confidently matched, 0 = confidently dark."""
    for c in contacts:
        if c.dist_m is None:
            c.match_score = 0.0
        else:
            c.match_score = max(0.0, 1.0 - c.dist_m / max_dist_m)
    return contacts


def separation_auc(contacts: list[MatchedContact], truth_matched: list[bool]) -> float:
    """ROC AUC of match_score against ground-truth matched/dark labels."""
    y_true = [1 if t else 0 for t in truth_matched]
    y_score = [c.match_score for c in contacts]
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/correlate tests/score -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/correlate src/septentrion/score tests/correlate tests/score
git commit -m "feat: add spatial correlation and darkness score"
```

---

## Task 12: Walking-skeleton pipeline

**Files:**
- Create: `src/septentrion/pipelines/__init__.py`
- Create: `src/septentrion/pipelines/skeleton.py`
- Test: `tests/pipelines/test_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipelines/test_skeleton.py
import json
from pathlib import Path

from septentrion.config import Config
from septentrion.pipelines.skeleton import run_skeleton


def test_run_skeleton_writes_results_and_metrics(tmp_path: Path):
    cfg = Config(out_dir=str(tmp_path))
    result = run_skeleton(cfg)

    contacts_csv = tmp_path / "skeleton" / "contacts.csv"
    metrics_json = tmp_path / "skeleton" / "metrics.json"
    assert contacts_csv.exists()
    assert metrics_json.exists()
    # provenance sidecars exist
    assert (tmp_path / "skeleton" / "contacts.csv.prov.json").exists()

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert metrics["n_detections"] == result.n_detections
    assert metrics["n_matched"] + metrics["n_dark"] == result.n_detections
    assert 0.0 <= metrics["separation_auc"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipelines/test_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.pipelines'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/pipelines/__init__.py
```

```python
# src/septentrion/pipelines/skeleton.py
"""End-to-end synthetic run: generate -> detect -> correlate -> score -> report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from septentrion.config import Config
from septentrion.correlate.spatial import correlate
from septentrion.detect.stub import StubDetector
from septentrion.io.synthetic import generate_scene
from septentrion.logging_setup import setup_logging
from septentrion.provenance import _git_commit, write_artifact
from septentrion.rng import set_seed
from septentrion.score.darkness import score_contacts, separation_auc


@dataclass
class SkeletonResult:
    n_detections: int
    n_matched: int
    n_dark: int
    separation_auc: float


def run_skeleton(cfg: Config, scene_id: str = "syn-0") -> SkeletonResult:
    log = setup_logging(run_id=scene_id, git_commit=_git_commit())
    rng = set_seed(cfg.seed)

    scene = generate_scene(rng, cfg.synthetic, scene_id=scene_id)
    log.info("scene_generated", scene_id=scene_id, n_truth=len(scene.truth), n_ais=len(scene.ais))

    detections = StubDetector(threshold=cfg.detection.confidence).detect(scene.raster, scene_id)
    contacts = correlate(detections, scene.ais, max_dist_m=cfg.correlation.max_dist_m)
    contacts = score_contacts(contacts, max_dist_m=cfg.correlation.max_dist_m)

    # Evaluate against ground truth: a detection is truly matched if its pixel
    # coincides with a non-dark truth vessel.
    dark_pixels = {(d.row, d.col) for d in scene.truth if d.__dict__.get("label") == "dark"}
    truth_matched = [(c.detection.row, c.detection.col) not in dark_pixels for c in contacts]
    auc = separation_auc(contacts, truth_matched)

    n_matched = sum(1 for c in contacts if c.label == "matched")
    result = SkeletonResult(
        n_detections=len(contacts),
        n_matched=n_matched,
        n_dark=len(contacts) - n_matched,
        separation_auc=auc,
    )

    out = Path(cfg.out_dir) / "skeleton"
    header = "scene_id,row,col,lon,lat,label,match_score,mmsi,dist_m\n"
    rows = "".join(
        f"{c.detection.scene_id},{c.detection.row},{c.detection.col},"
        f"{c.detection.lon},{c.detection.lat},{c.label},{c.match_score:.4f},"
        f"{c.mmsi if c.mmsi is not None else ''},{c.dist_m if c.dist_m is not None else ''}\n"
        for c in contacts
    )
    write_artifact(out / "contacts.csv", (header + rows).encode(), cfg=cfg, inputs=[scene_id])
    metrics = {
        "n_detections": result.n_detections,
        "n_matched": result.n_matched,
        "n_dark": result.n_dark,
        "separation_auc": result.separation_auc,
    }
    write_artifact(
        out / "metrics.json", json.dumps(metrics, indent=2).encode(), cfg=cfg, inputs=[scene_id]
    )
    log.info("skeleton_done", **metrics)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipelines/test_skeleton.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/septentrion/pipelines tests/pipelines/test_skeleton.py
git commit -m "feat: add walking-skeleton pipeline"
```

---

## Task 13: CLI

**Files:**
- Create: `src/septentrion/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from pathlib import Path

from typer.testing import CliRunner

from septentrion.cli import app

runner = CliRunner()


def test_run_skeleton_command(tmp_path: Path):
    result = runner.invoke(app, ["run-skeleton", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "skeleton" / "metrics.json").exists()
    assert "separation_auc" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septentrion.cli'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/septentrion/cli.py
"""Command-line entry points for the PoC."""

from __future__ import annotations

import typer

from septentrion.config import Config
from septentrion.pipelines.skeleton import run_skeleton

app = typer.Typer(help="Septentrion Aperture PoC commands.")


@app.command("run-skeleton")
def run_skeleton_command(
    out_dir: str = typer.Option("outputs", help="Directory for results."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Run the synthetic walking skeleton end to end."""
    cfg = Config(out_dir=out_dir, seed=seed)
    result = run_skeleton(cfg)
    typer.echo(
        f"detections={result.n_detections} matched={result.n_matched} "
        f"dark={result.n_dark} separation_auc={result.separation_auc:.3f}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Full verification — lint, type, all tests, real CLI run**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest && uv run septentrion run-skeleton --out-dir outputs
```
Expected: ruff clean; ty success; all tests pass; CLI prints a line like `detections=40 matched=28 dark=12 separation_auc=1.000` and writes `outputs/skeleton/`.

- [ ] **Step 6: Commit**

```bash
git add src/septentrion/cli.py tests/test_cli.py
git commit -m "feat: add typer CLI with run-skeleton command"
```

---

## Self-Review

**Spec coverage (Stage 0 + Stage 1):**
- §5 stack (Python 3.14, uv, hatchling, ruff, ty, pytest, pydantic-settings, structlog) → Tasks 1–2.
- §4 cross-platform: torch optional extras (Task 1), 3-OS CI matrix (Task 2). ✓
- §6 modules: config/rng/provenance/logging (Tasks 3–6), schemas (Task 7), geo (Task 8), io/synthetic (Task 9), detect (Task 10), correlate/score (Task 11), pipelines (Task 12), cli (Task 13). ✓
- §6 walking skeleton (synthetic end-to-end, CPU, no torch/GPU) → Task 12. ✓
- §8 provenance (`.prov.json`, seed, git commit, config dump, versions) → Task 5, used in Task 12. ✓
- §9 testing (markers, no large files, determinism asserted on shapes/values) → pytest config (Task 1), synthetic fixtures throughout. ✓
- Deferred to later plans (correctly out of scope here): real xView3/AIS loaders + CRS reprojection (Plan 2/4), spatiotemporal correlation (Plan 2), PR/ROC figures (Plan 3), real detector (Plan 4), E4 graph + reporting (Plan 5).

**Placeholder scan:** No TBD/TODO. The one prose note (Task 9 `__dict__["label"]`) documents a real, working mechanism used in Task 12 — not a placeholder.

**Type consistency:** `Detection(scene_id,row,col,lon,lat,detection_confidence)`, `AisRecord(mmsi,timestamp,lon,lat)`, `MatchedContact(detection,label,match_score,mmsi,dist_m,dt_s)` consistent across Tasks 7/9/10/11/12. `correlate(detections, ais, max_dist_m)`, `score_contacts(contacts, max_dist_m)`, `separation_auc(contacts, truth_matched)`, `StubDetector(threshold).detect(raster, scene_id)`, `generate_scene(rng, cfg, scene_id)`, `Config(out_dir, seed)` used consistently. ✓
