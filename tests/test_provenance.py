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
