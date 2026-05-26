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
