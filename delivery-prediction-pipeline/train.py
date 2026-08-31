"""Train a delivery-time regression model and optionally publish it to Hugging Face."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_CANDIDATES = ["delivery_time_minutes", "delivery_time", "time_taken", "Time_taken(min)", "Delivery_Time", "delivery_duration_minutes"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the source CSV.")
    parser.add_argument("--target", help="Target column; inferred when omitted.")
    parser.add_argument("--model-dir", default="models/delivery-time-regressor")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hub-repo", help="Optional model repo, e.g. username/delivery-time-regressor.")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()

def choose_target(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Target '{requested}' is not a CSV column. Available: {list(frame.columns)}")
        return requested
    columns = {column.lower(): column for column in frame.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    raise ValueError("Could not infer target; pass --target COLUMN_NAME.")

def clean_target(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(series, errors="coerce")

def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    target = choose_target(frame, args.target)
    frame[target] = clean_target(frame[target])
    frame = frame.dropna(subset=[target]).copy()
    if len(frame) < 10:
        raise ValueError("At least 10 rows with a valid target are required.")
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], verbose_feature_names_out=False)
    model = Pipeline([("preprocess", preprocessor), ("regressor", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=args.random_state))])
    X_train, X_test, y_train, y_test = train_test_split(features, frame[target], test_size=args.test_size, random_state=args.random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {"mae_minutes": round(float(mean_absolute_error(y_test, predictions)), 4), "rmse_minutes": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4), "r2": round(float(r2_score(y_test, predictions)), 4), "train_rows": len(X_train), "test_rows": len(X_test)}
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    metadata = {"target": target, "feature_columns": features.columns.tolist(), "numeric_features": numeric, "categorical_features": categorical, "metrics": metrics}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "README.md").write_text("---\nlibrary_name: scikit-learn\ntags:\n- tabular-regression\n- delivery-prediction\n---\n\n# Delivery Time Regressor\n\nA scikit-learn model that predicts delivery time in minutes.\n\n## Evaluation\n\n" + "\n".join(f"- {key}: {value}" for key, value in metrics.items()) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if args.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(args.hub_repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.hub_repo, repo_type="model", folder_path=str(model_dir), commit_message="Upload delivery-time regressor")
        print(f"Published model to https://huggingface.co/{args.hub_repo}")

if __name__ == "__main__":
    main()
