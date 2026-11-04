# BLFO Bitumen Prediction MVP

This repository currently contains a first executable machine-learning slice for
predicting modified-bitumen properties. It does not contain customer production
data or a pre-trained model. The API intentionally refuses predictions until a
model has been trained with a real, validated dataset.

## What is implemented

- A scikit-learn random-forest pipeline for penetration, softening point,
  viscosity at 135 C, and elastic recovery.
- A CSV training command with a held-out evaluation set. If `batch_id` is
  supplied, rows from the same batch are kept together during the split.
- Per-target RMSE, MAE, and R-squared metrics, plus nominal 95% split-conformal
  prediction intervals calibrated only on the held-out records.
- A FastAPI endpoint that returns a prediction ID, model version, intervals,
  and required laboratory confirmation tests.
- A local JSON Lines prediction audit log.

Performance metrics from this pipeline are not known until it is trained and
evaluated on representative historical data. A random split without `batch_id`
does not establish generalization to unseen production batches. Even grouped
holdout results need review by process and laboratory experts before operational
use. Predictions are decision support and do not replace laboratory testing.

PG grade is returned as `null`: penetration, softening point, and viscosity
alone are not sufficient to determine a standards-compliant performance grade.
The lubricant optimizer is currently a non-operational demonstration using
synthetic component properties and a simplified linear blending approximation.
Its results must not be used for production or compliance decisions.

## Dataset format

Provide a UTF-8 CSV with one row per completed lab-tested blend. Required
columns:

```text
base_penetration,base_softening_point,base_viscosity_135c,
polymer_material_id,polymer_percentage,mixing_temperature,
mixing_time_minutes,mixing_speed_rpm,
output_penetration,output_softening_point,output_viscosity_135c,
output_elastic_recovery
```

An optional `batch_id` column enables a group-based holdout split and is
recommended. All required feature and target values must be numeric and
complete. The trainer requires at least 19 holdout records to calibrate the
nominal 95% intervals; with the default 20% split this means at least 95
complete records. Inspect and clean units, duplicate records, outliers, and
measurement provenance before training. Do not mix units or test methods.

## Train and evaluate

From `backend/`:

```powershell
python -m pip install -r requirements.txt
python -m app.train_model --input data/bitumen_history.csv
```

The command writes `artifacts/bitumen_model.joblib` and
`artifacts/bitumen_model.metrics.json`. It prints the same evaluation report
and exits with an error if the dataset is missing, malformed, or too small.
Neither dataset nor generated model artifact should be committed to source
control.

## Run the API

From `backend/`, after training:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Set `BLFO_MODEL_PATH` to use another model artifact and
`BLFO_PREDICTION_LOG` to change the local audit-log path. The default log is
`artifacts/predictions.jsonl`.

`POST /api/v1/bitumen/predict` accepts the base properties, polymer material
and percentage, and mixing temperature, time, and speed. `GET /health` checks
that the model artifact can be loaded and has valid prediction metadata; it
returns the loaded model version or an explicit readiness error. Prediction
requests return HTTP 503 when no trained artifact is available.

## Lubricant optimizer demo

`GET /api/v1/lubricant/catalog` shows a fixed synthetic component catalog.
`POST /api/v1/lubricant/optimize` minimizes the demo cost subject to requested
KV100, viscosity-index, pour-point, batch-size, and stock limits. The response
includes `data_source: "synthetic_demo"`, `operational_use_allowed: false`, and
warnings. This is only an API/optimization workflow prototype: the invented
values and linear blending approximation are not validated lubricant science.
Replace them with traceable laboratory measurements and validated blending
rules before any operational use. The public
[lubricant-formulation-dataset](https://github.com/iliana-vargas/lubricant-formulation-dataset)
is a small reference catalog, not a verified formulation experiment dataset,
so it is not used to train or calibrate this optimizer.

## Tests

From `backend/`:

```powershell
python -m pytest
```

Tests use generated data only to verify code paths and schema behavior. Their
results are not product accuracy evidence.
