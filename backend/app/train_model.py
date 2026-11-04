from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the BLFO bitumen model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the historical-data CSV.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/bitumen_model.joblib"),
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/bitumen_model.metrics.json"),
        help="Output path for the holdout evaluation report.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    report = train_model(
        args.input,
        args.artifact,
        args.report,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
