# Septentrion Aperture — Proof-of-Concept Design

**Date:** 2026-05-25
**Repo:** `github.com/Lucention/septentrion` · codename *Septentrion Aperture*
**Bid:** IDEaS CFP006 · Challenge 13 — Multi-modal AI for Advanced Situational Decisions · Solicitation W7714-248676/013
**Component:** 1a (Conceive) · TRL 1–3 · ≤ 6 months · ≤ CAD 250,000
**Submission closes:** 2 June 2026, 14:00 EDT

---

## 1. Purpose

This repository holds the **proof-of-concept (PoC)** that substantiates the **TRL 3** claim in the Component 1a bid for Arctic Maritime Domain Awareness. The PoC is the single open item blocking submission: its results fill the blank in proposal box **MC-1** ("number of scenes processed, the correlation/match rate achieved, and the key finding") and are referenced in **PRC-1**.

The PoC must produce **evidence** — scripts/notebooks, figures, and numbers — demonstrating that the *separate elements* of the cross-modal fusion technology work on real open data. It is **not** a polished, integrated product.

### TRL-3 guardrail (governing constraint)

> TRL 3 = experimental proof-of-concept of the **separate elements**, not an integrated breadboard. Integration of the elements would constitute TRL 4 and would **contradict** the bid. Every design decision below is bounded by this: build demonstrative evidence per element, not a unified pipeline product. Optional elements stay as single demonstrative examples.

---

## 2. Scope

### In scope — the required evidence elements

- **E1 — SAR vessel detection.** Run an existing/baseline detector over ~10–20 Sentinel-1 scenes; output detections + detection confidence. Reuse pretrained weights (xView3 baseline or a top open-sourced solution); **no training from scratch.**
- **E2 — Spatiotemporal SAR-to-AIS correlation.** For ≥1 scene, ingest real AIS for the matching window, project to a common frame, and match detections to AIS within distance/time tolerance; label each detection matched or dark with a match score.
- **E3 — Confidence / darkness score.** Produce a score separating matched from dark detections; show separation via a PR/ROC curve, distribution separation, or match-rate statistics.

### In scope — one optional stub (kept as a single demonstrative example)

- **E4 — Entity-resolution graph stub.** Link one object across two revisits using a graph. Single example only — not a persistent multi-object tracker.

### Explicitly out of scope

- **E5 (LLM explanation stub)** — deferred; not part of this PoC.
- Any **integration** of E1–E4 into a unified pipeline (would imply TRL 4).
- Model training; production hardening; UI; deployment automation.
- Polar-AIS coverage solutions — sparsity is **reported as a finding**, not engineered around.

---

## 3. Success criteria

The PoC is done when it can produce, reproducibly, the following for MC-1/PRC-1:

1. **Count of Sentinel-1 scenes processed** (target ~10–20 for E1; ≥1 for E2/E3).
2. **SAR-to-AIS match/correlation rate** and matched-vs-dark detection counts.
3. **A separation metric** (PR/ROC curve or distribution separation) for the confidence/darkness score.
4. **A one-line headline result** suitable to drop verbatim into MC-1.
5. **One E4 example** linking an object across two revisits.

Every figure and number must be traceable back to its inputs, parameters, seed, code commit, and locked environment (see §8, Provenance).

---

## 4. Cross-platform compliance (governing constraint)

**Every package and tool used in this project MUST be installable and functional on all three platforms — Windows, Linux, and macOS — unless a specific dependency is expressly documented in this spec as platform-limited.** This is a hard requirement on dependency selection: before any new dependency is added, confirm it ships working wheels/binaries for Windows-x64, Linux (x86_64 + arm64), and macOS-arm64 on the project's pinned Python version. A dependency that cannot meet this bar is rejected unless an explicit exception is recorded in the table below.

### Verified platform support (Python 3.14, as of 2026-05-25)

All core dependencies ship `cp314` wheels (or are pure-Python) for Windows-x64, Linux x86_64/arm64, and macOS-arm64: `numpy`, `pandas`, `scikit-learn`, `rasterio`, `shapely`, `pyproj`, `pyogrio`, `geopandas` (pure-Python), `networkx` (pure-Python). Tooling (`uv`, `ruff`, `ty`) are cross-platform Rust binaries.

### Documented platform exceptions

| Dependency | Limitation | Rationale / mitigation |
|---|---|---|
| `torch`, `torchvision` | macOS wheels are **arm64 only** — no Intel-Mac (x86_64) wheel | `torch` is an **optional extra** used only by E1 (detection). The novel CPU fusion layer (E2/E3/E4) runs with no torch installed, so Intel Macs retain full E2–E4 capability. Apple Silicon, Linux, and Windows get full coverage. |
| CUDA torch (`cu*` extra) | GPU wheels are **Linux/Windows only**; macOS uses CPU/MPS | The `cpu` extra works on all platforms; the `cu*` extra is gated by platform marker to Linux/Windows. GPU is needed only for the detector *run*, never for dev/test. |

Any future dependency that violates the three-platform rule must be added to this table with an explicit rationale, or it must not be used.

---

## 5. Engineering stack

Decided from a survey of current (early 2026) Python best practices. Rationale summarized; see decision log for the contested calls.

| Concern | Decision | Notes |
|---|---|---|
| **Python** | **3.14** (`requires-python = ">=3.14"`) | Latest stable; full wheel coverage confirmed for the whole stack. |
| Layout / packaging | `src/` layout, installable package `septentrion`, **hatchling** backend (+`hatch-vcs` for git-derived version) | `src/` forces testing the *installed* package; git-derived version aids provenance. |
| Deps & env | **uv** + committed `uv.lock`; runtime deps in `[project.dependencies]`, tooling in PEP 735 `[dependency-groups]`, **torch as optional `cpu`/`cu*` extra** | One lockfile resolves correct torch per platform. |
| Geospatial | wheels via uv: `rasterio`, `shapely`, `pyproj`, `geopandas`, `pyogrio` — **never depend on `gdal` directly** | rasterio bundles GDAL; `gdal` has no wheels. conda not needed. |
| Lint / format | **ruff** (curated rules, not `ALL`; `NPY`/`PD` on; docstrings/annotations off) | Pragmatic for research code. |
| Type checking | **`ty`** (Astral; Beta as of Dec 2025) as the gate | Chosen for Astral-stack coherence + the gradual guarantee (annotating valid code never adds errors), which suits mixed typed-library/loose-notebook code. Geo-lib stub gaps handled by ignore-missing-imports — same as mypy. Fallback: mypy is a one-line dev-dep swap if `ty` blocks us. |
| Tests | **pytest**, synthetic tiny fixtures, `gpu`/`slow`/`integration` markers skipped by default | Suite runs with no GPU and no large data. |
| Config | **pydantic-settings + TOML**, one model per stage; `model_dump()` feeds provenance | Hydra is overkill for a 4-stage PoC. |
| Reproducibility | `set_seed()` + `np.random.default_rng`; **`.prov.json` sidecar** per artifact + committed `uv.lock` + git commit | The provenance chain is itself on-message for the bid's "provenance-tracked" thesis. |
| Notebooks | **jupytext `py:percent`** committed, `.ipynb` git-ignored, papermill-executed in CI | Reviewable diffs; notebooks proven to run. |
| Logging | **structlog** over stdlib: JSON lines to file + console | Binds `run_id`/`scene_id` per stage. |
| CI / hooks | GitHub Actions (lint, format-check, `ty`, fast tests) on a **Windows + Linux + macOS** matrix; pre-commit (ruff, large-file block, nbstripout) | CI matrix enforces §4. |

---

## 6. Architecture

### Package layout

```
src/septentrion/
  config.py        pydantic-settings Config (nested per-stage models)
  rng.py           set_seed() -> np.random.Generator
  provenance.py    write_artifact() + .prov.json sidecar
  logging_setup.py structlog configuration
  schemas.py       data contracts: Detection, AisRecord, MatchedContact, VesselTrack
  io/              xView3 loader · AIS loader (NOAA CSV / GFW) · synthetic-sample generator
  geo/             CRS transforms, project-to-common-frame, windowed raster reads
  detect/          E1: detector interface + stub detector + pretrained xView3 wrapper
  correlate/       E2: spatiotemporal SAR<->AIS association (matched vs dark)
  score/           E3: confidence/darkness score + PR/ROC + separation metrics
  graph/           E4: entity-resolution stub (networkx), link one object across 2 revisits
  pipelines/       per-element orchestration the CLI/notebooks call
  cli.py           typer entry points (ingest/detect/correlate/score/graph)
tests/             mirrors src/, with tiny synthetic fixtures
notebooks/         jupytext py:percent — evidence figures (import septentrion)
scripts/           one-off data downloaders (xView3 slice, AIS)
data/              git-ignored — raw/interim inputs
outputs/           git-ignored — figures, numbers, .prov.json sidecars, run logs
```

Each subpackage has one clear purpose, a typed interface (`schemas.py` contracts), and is independently testable. The top-level `__init__.py` stays light (no heavy imports at package import time).

### Data contracts (`schemas.py`)

Defined as typed models so every stage's input/output is explicit and validated:

- **`Detection`** — `scene_id`, pixel/geo position, `detection_confidence`, bbox/size.
- **`AisRecord`** — `mmsi`, timestamp, lat/lon, kinematics.
- **`MatchedContact`** — a `Detection` plus match label (`matched` | `dark`), `match_score`, matched `mmsi` (if any), time/distance deltas.
- **`VesselTrack`** — an entity-resolution node linking detections/AIS across revisits (E4).

### Data flow (per element, NOT integrated)

```
ingest ──► E1 detect ──► detections
   │                         │
   │   (dev path: use xView3 ground-truth detections to build E2/E3 without the detector)
   ▼                         ▼
 AIS  ──────────────► E2 correlate ──► matched/dark + match_score ──► E3 score ──► metrics+figures
                                              │
                                              └──► E4 graph (one object, two revisits)
```

### Walking skeleton (de-risking strategy)

Day one: a **synthetic-sample generator** produces a tiny fake Sentinel-1 raster + fake AIS, and a **stub detector** emits fake detections. This pushes data end-to-end through ingest → E2 → E3 → E4 on **CPU, any platform, no real data, no GPU, no model weights** — making the whole pipeline runnable and unit-testable immediately. Real data and the real detector swap in behind the same interfaces.

A second de-risking lever: because xView3-SAR ships AIS-matched and dark **ground-truth detections**, E2/E3 can be developed and validated against those provided detections **before E1 (the detector) exists** — front-loading the novel, CPU-only fusion layer.

---

## 7. Data sources & ingest

| Modality | Primary source | Notes |
|---|---|---|
| Sentinel-1 GRD + labels | **xView3-SAR** dataset (Sentinel-1 + AIS-derived labels, dark ground truth built in) | 991 scenes, 243k labeled objects, VV+VH, per-label HIGH/MEDIUM/LOW confidence. Credentialed access at `iuu.xview.us`. Raw Sentinel-1 also via Copernicus Data Space Ecosystem (CDSE). Confirm endpoints before use. |
| Sentinel-1 (tiny entry point) | **SARFish sample** (~4–8 GB) — xView3-derived, adds complex/SLC | Recommended first real data for the walking skeleton: small enough to pull immediately, same label schema as xView3 (carries the xView3 community license; sourced from `iuu.xview.us`). Scale up to full xView3 once the pipeline runs. |
| AIS | **NOAA Marine Cadastre** (open CSV, US EEZ only) or **Global Fishing Watch API** (open token, global) | Demonstrate mechanism in a well-covered region; report polar sparsity as a finding. |
| Sentinel-2 (optional) | CDSE | Not in PoC scope unless trivially available. |
| Synthetic | generated locally | Walking-skeleton fallback; ships as tiny test fixtures. |

**Licensing note:** xView3-SAR (and the SARFish derivative) are distributed under a **credentialed "community" license — not an OSI-open license**. This is acceptable for the PoC (open-source-available evidence; no redistribution of the data), but the proposal's "open-source means only" constraint refers to *reference availability*, which these satisfy. Confirm exact license terms at `iuu.xview.us` before any redistribution.

**Arctic domain-shift limitation (a finding, not a fix):** xView3/SARFish scenes target global *fishing* hotspots, **not polar waters**, and terrestrial AIS is sparse in the high Arctic (dense polar coverage needs satellite-AIS). A detector trained on this data will produce **sea-ice false positives** and run against thin AIS ground truth in the Arctic. Consistent with the bid's TRL-3 framing, the PoC **demonstrates the fusion mechanism in well-covered regions and reports sea-ice clutter and polar-AIS sparsity as explicit findings** — it does not attempt Arctic domain adaptation (that is follow-on-component work).

**Raster handling gotchas (baked into `io/`/`geo/`):** xView3 scenes are ~29k×24k px UTM GeoTIFFs (VV+VH ~GB each) — use **rasterio windowed reads**, never full-scene loads; one dataset handle per worker (GDAL handles aren't thread-safe); ancillary rasters are on a different grid (resample, don't assume pixel alignment); SAR backscatter is in dB and 0 ≠ nodata (read masks explicitly); each scene has its own UTM CRS (reproject AIS into the scene CRS, cache the transformer).

The clean machine starts with **nothing downloaded**: `scripts/` provides documented downloaders, and the synthetic generator makes the code runnable before any real data lands.

---

## 8. Reproducibility & provenance

The provenance chain is a deliverable, not a nicety (it backs the bid's "provenance-tracked" claim and lets a reviewer re-derive any number):

- **Config** (`pydantic-settings` + `config.toml`) is validated and `model_dump()`-ed into every output's metadata.
- **Seeds** set via one `set_seed()` (torch global + `use_deterministic_algorithms` + a returned `np.random.default_rng` passed through stages); seed recorded in provenance.
- **Provenance sidecar:** every artifact writes a `<artifact>.prov.json` capturing input scene IDs, resolved config, seed, git commit (with dirty flag), Python/platform, and key library versions.
- **Environment** pinned by committed `uv.lock`. Chain: figure → `.prov.json` → git commit → `uv.lock` → exact bytes.
- **Data versioning:** raw data stays **out of git** (documented fetch + recorded scene IDs); `git-lfs` only if a small binary must travel with the repo; **DVC is out of scope** (overkill for a 1-week PoC).

---

## 9. Testing strategy

- `pytest` with `src/` layout; tests mirror `src/septentrion/`.
- **Synthetic tiny fixtures** (small in-memory rasters/AIS, seeded) — no large files committed.
- Markers `gpu` / `slow` / `integration` **skipped by default**, gated on dependency availability (torch/GDAL presence) so the fast suite runs everywhere with no GPU.
- Determinism asserted on shapes/dtypes/`allclose`, not exact floats.
- CI runs the fast suite on the **three-OS matrix**; GPU/slow tests run on a scheduled job (no GPU on PR runners).

---

## 10. Build order

Sequenced to front-load the novel, CPU-only fusion work and defer the heavy GPU detector. Maps to the proposal's four milestones but each stage delivers standalone evidence (not integration).

| # | Stage | Depends on | Compute |
|---|---|---|---|
| 0 | Repo scaffolding + Python tooling (uv, ruff, ty, pytest, pre-commit, CI, config/rng/provenance/logging) | — | any |
| 1 | Data contracts + synthetic sample + ingest (xView3 + AIS loaders); walking skeleton runs | 0 | CPU |
| 2 | **E2** — SAR↔AIS spatiotemporal correlation (matched vs dark) | 1 | CPU |
| 3 | **E3** — confidence/darkness score + PR/ROC + separation figures | 2 | CPU |
| 4 | **E1** — pretrained detector integration (wrapper over xView3 baseline) | 1 | GPU |
| 5 | **E4** — entity-resolution graph stub (one object, two revisits) | 2 | CPU |
| 6 | Reporting — numbers + figures + provenance into MC-1/PRC-1 | 3,4,5 | CPU |

Stages 2–3 yield a real result on real data (using xView3's own detections) before the detector exists — the fastest path to filling the MC-1 blank within the ~8-day window.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Timeline** — ~8 days to 2 June | Walking skeleton + xView3 ground-truth detections let E2/E3 produce numbers before E1; E1 and E4 are parallelizable; E5 explicitly dropped. |
| **Data access** (CDSE/GFW endpoints, xView3 size) | Synthetic fallback keeps dev unblocked; downloaders scripted; confirm endpoints first. |
| **GPU availability** for E1 | Detector deferred to stage 4; everything else is CPU; torch optional. |
| **Cross-platform breakage** | §4 compliance rule + three-OS CI matrix; torch exceptions documented. |
| **TRL-3 overreach** (accidental integration → looks like TRL 4) | Per-element orchestration only; E4/E5 kept as single examples; no unified pipeline. |
| **`ty` beta churn** | mypy fallback is a one-line dev-dep swap. |
| **Polar AIS sparsity** | Demonstrate in well-covered region; report sparsity as a finding (not engineered around). |

---

## 12. Out of scope (YAGNI)

E5 LLM explanation; model training; element integration into a unified pipeline; UI; deployment automation; DVC; conda; supporting Python < 3.14; Intel-Mac GPU detection.
