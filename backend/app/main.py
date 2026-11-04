from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import joblib
from fastapi import FastAPI, HTTPException

from app.ml.bitumen import predict_with_bundle
from app.schemas.prediction import (
    BitumenPredictionRequest,
    BitumenPredictionResponse,
    HealthResponse,
)

app = FastAPI(title="BLFO Bitumen Prediction API", version="0.1.0")
_audit_lock = threading.Lock()


def _model_path() -> Path:
    return Path(os.getenv("BLFO_MODEL_PATH", "artifacts/bitumen_model.joblib"))


@lru_cache(maxsize=1)
def _load_model(path: str) -> dict:
    return joblib.load(path)


def _write_audit_record(record: dict) -> None:
    log_path = Path(os.getenv("BLFO_PREDICTION_LOG", "artifacts/predictions.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _audit_lock:
        with log_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_ready=_model_path().is_file())


@app.post("/api/v1/bitumen/predict", response_model=BitumenPredictionResponse)
def predict_bitumen(request: BitumenPredictionRequest) -> BitumenPredictionResponse:
    model_path = _model_path()
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="No trained bitumen model is available. Train a model on validated historical data first.",
        )

    try:
        bundle = _load_model(str(model_path.resolve()))
        features = {
            "base_penetration": request.base_properties.penetration,
            "base_softening_point": request.base_properties.softening_point,
            "base_viscosity_135c": request.base_properties.viscosity_135c,
            "polymer_material_id": request.polymer.material_id,
            "polymer_percentage": request.polymer.percentage,
            "mixing_temperature": request.mixing_temperature,
            "mixing_time_minutes": request.mixing_time_minutes,
            "mixing_speed_rpm": request.mixing_speed_rpm,
        }
        prediction = predict_with_bundle(bundle, features)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=500, detail=f"Unable to use the trained model: {error}") from error

    prediction_id = uuid4()
    response = BitumenPredictionResponse(
        **prediction,
        required_tests=[
            "Penetration (ASTM D5)",
            "Softening point (ASTM D36)",
            "Viscosity at specified temperatures",
            "Elastic recovery (if polymer-modified)",
        ],
        prediction_id=prediction_id,
    )
    _write_audit_record(
        {
            "prediction_id": str(prediction_id),
            "model_version": response.model_version,
            "input": request.model_dump(),
            "output": response.model_dump(mode="json"),
        }
    )
    return response
