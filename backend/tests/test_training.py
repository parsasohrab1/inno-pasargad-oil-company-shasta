import numpy as np
import pandas as pd
import pytest

from app.ml.bitumen import FEATURE_COLUMNS, TARGET_COLUMNS, validate_dataset
from app.ml.training import conformal_radius, train_model


def make_dataset(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "base_penetration": rng.uniform(30, 90, rows),
        "base_softening_point": rng.uniform(40, 60, rows),
        "base_viscosity_135c": rng.uniform(0.1, 0.8, rows),
        "polymer_material_id": rng.integers(1, 4, rows),
        "polymer_percentage": rng.uniform(0, 6, rows),
        "mixing_temperature": rng.uniform(150, 190, rows),
        "mixing_time_minutes": rng.uniform(30, 180, rows),
        "mixing_speed_rpm": rng.uniform(100, 800, rows),
        "batch_id": [f"batch-{index // 4}" for index in range(rows)],
    }
    data["output_penetration"] = data["base_penetration"] - 2 * data["polymer_percentage"]
    data["output_softening_point"] = (
        data["base_softening_point"] + 1.5 * data["polymer_percentage"]
    )
    data["output_viscosity_135c"] = (
        data["base_viscosity_135c"] + 0.02 * data["polymer_percentage"]
    )
    data["output_elastic_recovery"] = np.clip(
        10 + 12 * data["polymer_percentage"], 0, 100
    )
    return pd.DataFrame(data)


def test_training_produces_model_and_holdout_metrics(tmp_path):
    dataset = tmp_path / "history.csv"
    artifact = tmp_path / "model.joblib"
    report_path = tmp_path / "metrics.json"
    make_dataset().to_csv(dataset, index=False)

    report = train_model(dataset, artifact, report_path)

    assert artifact.is_file()
    assert report_path.is_file()
    assert report["split_method"] == "grouped_by_batch_id"
    assert report["holdout_rows"] >= 19
    assert set(report["metrics"]) == {
        "penetration",
        "softening_point",
        "viscosity_135c",
        "elastic_recovery",
    }
    assert all(metric["rmse"] >= 0 for metric in report["metrics"].values())
    assert all(radius >= 0 for radius in report["interval_radii"].values())


def test_dataset_rejects_missing_or_invalid_values():
    frame = make_dataset()
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(frame.drop(columns=[TARGET_COLUMNS[0]]))

    frame.loc[0, FEATURE_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError, match="numeric, non-empty"):
        validate_dataset(frame)


def test_conformal_radius_uses_finite_sample_quantile():
    assert conformal_radius(np.arange(1, 20, dtype=float)) == 19


def test_training_rejects_too_small_calibration_set(tmp_path):
    dataset = tmp_path / "small.csv"
    make_dataset(50).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="At least 19 holdout rows"):
        train_model(dataset, tmp_path / "model.joblib", tmp_path / "metrics.json")
