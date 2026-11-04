from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.bitumen import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TARGET_NAMES,
    make_model,
    validate_dataset,
)

CONFIDENCE_LEVEL = 0.95
MIN_CALIBRATION_ROWS = math.ceil(1 / (1 - CONFIDENCE_LEVEL)) - 1


def _split_indices(frame: pd.DataFrame, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(len(frame))
    if "batch_id" in frame and frame["batch_id"].nunique() >= 2:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_indices, test_indices = next(
            splitter.split(indices, groups=frame["batch_id"].astype(str))
        )
        return train_indices, test_indices, "grouped_by_batch_id"

    from sklearn.model_selection import train_test_split

    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    return train_indices, test_indices, "random_rows"


def conformal_radius(absolute_errors: np.ndarray, coverage: float = CONFIDENCE_LEVEL) -> float:
    if absolute_errors.ndim != 1 or len(absolute_errors) == 0:
        raise ValueError("Calibration errors must be a non-empty one-dimensional array.")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one.")
    rank = min(math.ceil((len(absolute_errors) + 1) * coverage), len(absolute_errors))
    return float(np.partition(absolute_errors, rank - 1)[rank - 1])


def train_model(
    input_path: Path,
    artifact_path: Path,
    report_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {input_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one.")

    raw = pd.read_csv(input_path)
    data = validate_dataset(raw)
    train_indices, test_indices, split_method = _split_indices(data, test_size, random_state)
    if len(test_indices) < MIN_CALIBRATION_ROWS:
        raise ValueError(
            f"At least {MIN_CALIBRATION_ROWS} holdout rows are required to calibrate "
            f"nominal 95% prediction intervals; got {len(test_indices)}. "
            "With the default 20% split, provide at least 95 complete rows."
        )
    if len(train_indices) < 2:
        raise ValueError("At least two training rows are required.")

    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    x_train = train.loc[:, FEATURE_COLUMNS]
    x_test = test.loc[:, FEATURE_COLUMNS]
    y_train = train.loc[:, TARGET_COLUMNS]
    y_test = test.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)

    estimator = make_model()
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    residuals = y_test - predictions

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics: dict[str, dict[str, float]] = {}
    interval_radii: dict[str, float] = {}
    for index, column in enumerate(TARGET_COLUMNS):
        target_errors = residuals[:, index]
        metrics[TARGET_NAMES[column]] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test[:, index], predictions[:, index]))),
            "mae": float(mean_absolute_error(y_test[:, index], predictions[:, index])),
            "r2": float(r2_score(y_test[:, index], predictions[:, index])),
        }
        interval_radii[column] = conformal_radius(np.abs(target_errors))

    dataset_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    model_version = f"bitumen-rf-{dataset_hash[:12]}-{random_state}"
    report: dict[str, Any] = {
        "model_version": model_version,
        "algorithm": "RandomForestRegressor",
        "dataset_sha256": dataset_hash,
        "split_method": split_method,
        "training_rows": int(len(train_indices)),
        "holdout_rows": int(len(test_indices)),
        "confidence_level": CONFIDENCE_LEVEL,
        "metrics": metrics,
        "interval_radii": {
            TARGET_NAMES[column]: radius for column, radius in interval_radii.items()
        },
        "warning": (
            "Metrics and intervals are based on the specified holdout split. "
            "They are not a guarantee of performance on future production data."
        ),
    }
    bundle = {
        "estimator": estimator,
        "model_version": model_version,
        "dataset_sha256": dataset_hash,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "interval_radii": interval_radii,
        "confidence_level": CONFIDENCE_LEVEL,
        "training_rows": int(len(train_indices)),
        "holdout_rows": int(len(test_indices)),
        "split_method": split_method,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
