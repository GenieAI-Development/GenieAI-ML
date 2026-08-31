"""Evaluate a saved delivery-time model on the held-out test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument("--model-dir", default="models/delivery-time-regressor")
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.test_data)
    target = metadata["target"]
    missing = [column for column in metadata["feature_columns"] if column not in frame]
    if missing:
        raise ValueError(f"Test data is missing model features: {missing}")
    y_true = frame[target]
    y_pred = model.predict(frame[metadata["feature_columns"]])
    metrics = {
        "mae_minutes": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse_minutes": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "test_rows": len(frame),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
