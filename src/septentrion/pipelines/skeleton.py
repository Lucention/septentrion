"""End-to-end synthetic run: generate -> detect -> correlate -> score -> report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
