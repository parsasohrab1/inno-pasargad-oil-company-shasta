import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_lubricant_catalog_is_marked_synthetic():
    with TestClient(app) as client:
        response = client.get("/api/v1/lubricant/catalog")

    assert response.status_code == 200
    result = response.json()
    assert result["data_source"] == "synthetic_demo"
    assert result["operational_use_allowed"] is False
    assert len(result["components"]) == 5


def test_lubricant_optimizer_returns_transparent_demo_result():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/lubricant/optimize",
            json={
                "target_properties": {
                    "viscosity_kv100_min_cst": 5.5,
                    "viscosity_kv100_max_cst": 7,
                    "viscosity_index_min": 110,
                    "pour_point_max_c": -25,
                },
                "batch_size_kg": 1000,
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["data_source"] == "synthetic_demo"
    assert result["operational_use_allowed"] is False
    assert result["estimated_cost"] > 0
    assert sum(item["percentage"] for item in result["formulation"]) == pytest.approx(
        100
    )
    assert result["predicted_properties"]["viscosity_index"] >= 110
    assert result["predicted_properties"]["pour_point_c"] <= -25
    assert len(result["warnings"]) >= 2


def test_lubricant_optimizer_reports_infeasible_target():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/lubricant/optimize",
            json={
                "target_properties": {
                    "viscosity_kv100_min_cst": 50,
                    "viscosity_kv100_max_cst": 60,
                    "viscosity_index_min": 200,
                    "pour_point_max_c": -70,
                },
            },
        )

    assert response.status_code == 422
    assert "No feasible formulation" in response.json()["detail"]
