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
