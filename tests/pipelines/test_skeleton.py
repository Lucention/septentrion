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
