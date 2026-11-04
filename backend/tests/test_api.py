import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _load_model


class FixedEstimator:
    def predict(self, frame):
        from app.ml.bitumen import FEATURE_COLUMNS

        assert tuple(frame.columns) == FEATURE_COLUMNS
        return np.array([[42.0, 54.0, 0.4, 65.0]])


@pytest.fixture
def client(monkeypatch, tmp_path):
    _load_model.cache_clear()
    monkeypatch.setenv("BLFO_MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("BLFO_PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    with TestClient(app) as test_client:
        yield test_client
    _load_model.cache_clear()


def payload():
    return {
        "base_properties": {
            "penetration": 60,
            "softening_point": 49,
            "viscosity_135c": 0.3,
        },
        "polymer": {"material_id": 1, "percentage": 3.5},
        "mixing_temperature": 180,
        "mixing_time_minutes": 120,
        "mixing_speed_rpm": 500,
    }


def test_predict_requires_trained_model(client):
    response = client.post("/api/v1/bitumen/predict", json=payload())
    assert response.status_code == 503


def test_predict_returns_model_outputs_intervals_and_audit_log(client, tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": FixedEstimator(),
            "model_version": "test-model-v1",
            "interval_radii": {
                "output_penetration": 2,
                "output_softening_point": 1,
                "output_viscosity_135c": 0.05,
                "output_elastic_recovery": 4,
            },
            "confidence_level": 0.95,
        },
        artifact,
    )
    monkeypatch.setenv("BLFO_MODEL_PATH", str(artifact))

    response = client.post("/api/v1/bitumen/predict", json=payload())

    assert response.status_code == 200
    result = response.json()
    assert result["softening_point"] == 54
    assert result["pg_grade"] is None
    assert result["confidence_intervals"]["softening_point"] == {
        "lower": 53,
        "upper": 55,
        "nominal_coverage": 0.95,
    }
    assert result["model_version"] == "test-model-v1"
    assert (tmp_path / "predictions.jsonl").is_file()
    assert "prediction_id" in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8")


def test_predict_rejects_invalid_input(client):
    invalid_payload = payload()
    invalid_payload["base_properties"]["penetration"] = -5
    response = client.post("/api/v1/bitumen/predict", json=invalid_payload)

    assert response.status_code == 422
