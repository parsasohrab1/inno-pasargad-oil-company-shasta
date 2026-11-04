from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

FEATURE_COLUMNS = (
    "base_penetration",
    "base_softening_point",
    "base_viscosity_135c",
    "polymer_material_id",
    "polymer_percentage",
    "mixing_temperature",
    "mixing_time_minutes",
    "mixing_speed_rpm",
)

TARGET_COLUMNS = (
    "output_penetration",
    "output_softening_point",
    "output_viscosity_135c",
    "output_elastic_recovery",
)

TARGET_NAMES = {
    "output_penetration": "penetration",
    "output_softening_point": "softening_point",
    "output_viscosity_135c": "viscosity_135c",
    "output_elastic_recovery": "elastic_recovery",
}

REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS


def build_feature_frame(values: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    frame["polymer_material_id"] = frame["polymer_material_id"].astype(str)
    return frame


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    clean = frame.loc[:, list(REQUIRED_COLUMNS) + (["batch_id"] if "batch_id" in frame else [])].copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required feature and target columns must contain numeric, non-empty values.")
    if not np.isfinite(clean.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Required feature and target values must be finite.")
    if (
        (clean["polymer_material_id"] < 0).any()
        or (clean["polymer_material_id"] % 1 != 0).any()
    ):
        raise ValueError("polymer_material_id values must be non-negative integers.")
    if ((clean["polymer_percentage"] < 0) | (clean["polymer_percentage"] > 100)).any():
        raise ValueError("polymer_percentage values must be between 0 and 100.")
    if (clean["mixing_time_minutes"] <= 0).any() or (clean["mixing_speed_rpm"] <= 0).any():
        raise ValueError("Mixing time and speed must be greater than zero.")

    clean["polymer_material_id"] = clean["polymer_material_id"].astype(int).astype(str)
    if "batch_id" in clean and clean["batch_id"].isna().any():
        raise ValueError("batch_id must be populated for every row when the column is provided.")
    return clean


def make_model() -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical = ["polymer_material_id"]
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("polymer", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    estimator = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def validate_model_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise ValueError("The model artifact must contain a model bundle.")

    estimator = bundle.get("estimator")
    radii = bundle.get("interval_radii")
    version = bundle.get("model_version")
    coverage = bundle.get("confidence_level")
    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("The model artifact has no usable estimator.")
    if not isinstance(radii, dict) or not isinstance(version, str) or not version.strip():
        raise ValueError("The model artifact is missing required model metadata.")
    if not isinstance(coverage, (int, float)) or not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("The model artifact has an invalid confidence level.")
    for column in TARGET_COLUMNS:
        radius = radii.get(column)
        if not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius < 0:
            raise ValueError(f"The model artifact has no valid interval for {column}.")
    return bundle


def predict_with_bundle(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    bundle = validate_model_bundle(bundle)
    estimator = bundle["estimator"]
    radii = bundle["interval_radii"]
    version = bundle["model_version"]

    predictions = np.asarray(estimator.predict(build_feature_frame(features)))
    if predictions.shape != (1, len(TARGET_COLUMNS)) or not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid prediction values.")
    prediction = predictions[0]
    outputs: dict[str, Any] = {}
    intervals: dict[str, dict[str, float]] = {}
    for column, value in zip(TARGET_COLUMNS, prediction, strict=True):
        name = TARGET_NAMES[column]
        radius = radii[column]
        outputs[name] = float(value)
        intervals[name] = {
            "lower": float(value - radius),
            "upper": float(value + radius),
            "nominal_coverage": float(bundle["confidence_level"]),
        }

    return {
        **outputs,
        "pg_grade": None,
        "confidence_intervals": intervals,
        "model_version": version,
    }
