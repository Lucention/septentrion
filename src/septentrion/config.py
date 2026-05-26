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
