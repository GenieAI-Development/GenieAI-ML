"""Download the public Kaggle Food Delivery Dataset into data/raw/."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

KAGGLE_DATASET_ID = "gauravmalik26/food-delivery-dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    downloaded_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    csv_files = list(downloaded_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset: {downloaded_dir}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for csv_file in csv_files:
        destination = output / csv_file.name
        shutil.copy2(csv_file, destination)
        copied_files.append(destination)
    train_file = next((path for path in copied_files if path.name.lower() == "train.csv"), copied_files[0])
    print(f"Downloaded {KAGGLE_DATASET_ID}")
    print(f"Raw CSV files saved to: {output.resolve()}")
    print(f"Use this preparation input: {train_file}")


if __name__ == "__main__":
    main()
