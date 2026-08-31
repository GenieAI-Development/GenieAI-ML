"""Clean a delivery CSV, engineer useful features, and create train/test files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_NAMES = {
    "delivery_time_minutes", "delivery_time", "time_taken", "time_taken_min",
    "delivery_duration_minutes",
}


def snake_case(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def numeric_minutes(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(series, errors="coerce")


def add_distance(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = [
        ("restaurant_latitude", "restaurant_longitude", "delivery_location_latitude", "delivery_location_longitude"),
        ("store_latitude", "store_longitude", "drop_latitude", "drop_longitude"),
    ]
    for lat1, lon1, lat2, lon2 in aliases:
        if all(column in frame for column in (lat1, lon1, lat2, lon2)):
            values = frame[[lat1, lon1, lat2, lon2]].apply(pd.to_numeric, errors="coerce")
            p1, p2 = np.radians(values[lat1]), np.radians(values[lat2])
            dp = np.radians(values[lat2] - values[lat1])
            dl = np.radians(values[lon2] - values[lon1])
            a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
            frame["distance_km"] = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a.clip(upper=1)))
            break
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/train.csv", help="Raw CSV path created by download_dataset.py.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--target", help="Raw or normalized target column name.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    source = args.input
    frame = pd.read_csv(args.input)
    frame.columns = [snake_case(column) for column in frame.columns]
    requested = snake_case(args.target) if args.target else None
    target = requested or next((name for name in TARGET_NAMES if name in frame.columns), None)
    if not target or target not in frame:
        raise ValueError(f"Target not found. Use --target. Columns: {list(frame.columns)}")
    frame = frame.rename(columns={target: "delivery_time_minutes"})
    frame["delivery_time_minutes"] = numeric_minutes(frame["delivery_time_minutes"])
    frame = add_distance(frame)

    for date_column in ("order_date", "date", "created_at"):
        if date_column in frame:
            parsed = pd.to_datetime(frame[date_column], errors="coerce")
            frame["day_of_week"] = parsed.dt.day_name()
            frame["month"] = parsed.dt.month
            frame = frame.drop(columns=[date_column])
            break
    for time_column in ("time_orderd", "time_ordered", "order_time"):
        if time_column in frame:
            parsed = pd.to_datetime(frame[time_column], errors="coerce")
            frame["hour_of_day"] = parsed.dt.hour
            frame = frame.drop(columns=[time_column])
            break

    frame = frame.dropna(subset=["delivery_time_minutes"])
    frame = frame[(frame["delivery_time_minutes"] > 0) & (frame["delivery_time_minutes"] <= frame["delivery_time_minutes"].quantile(0.995))]
    id_columns = [column for column in frame if column == "id" or column.endswith("_id")]
    frame = frame.drop(columns=id_columns).drop_duplicates()
    if len(frame) < 20:
        raise ValueError("At least 20 valid rows are required after cleaning.")

    train, test = train_test_split(frame, test_size=args.test_size, random_state=args.random_state)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    summary = {"source": source, "train_rows": len(train), "test_rows": len(test), "features": [c for c in frame if c != "delivery_time_minutes"], "target": "delivery_time_minutes"}
    (output / "schema.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
