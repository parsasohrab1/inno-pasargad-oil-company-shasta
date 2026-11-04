from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class BitumenBaseProperties(BaseModel):
    penetration: float = Field(gt=0)
    softening_point: float = Field(gt=0)
    viscosity_135c: float = Field(gt=0)


class PolymerAdditive(BaseModel):
    material_id: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class BitumenPredictionRequest(BaseModel):
    base_properties: BitumenBaseProperties
    polymer: PolymerAdditive
    mixing_temperature: float = Field(gt=0)
    mixing_time_minutes: float = Field(gt=0)
    mixing_speed_rpm: float = Field(gt=0)


class PredictionInterval(BaseModel):
    lower: float
    upper: float
    nominal_coverage: float


class BitumenPredictionResponse(BaseModel):
    penetration: float
    softening_point: float
    viscosity_135c: float
    elastic_recovery: float
    pg_grade: None = None
    confidence_intervals: dict[str, PredictionInterval]
    required_tests: list[str]
    prediction_id: UUID
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_ready: bool
