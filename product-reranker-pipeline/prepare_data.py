"""Download public ESCI data and prepare grouped CrossEncoder training pairs."""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "tasksource/esci"
LABEL_SCORES = {
    "E": 1.0,
    "Exact": 1.0,
    "S": 0.7,
    "Substitute": 0.7,
    "C": 0.35,
    "Complement": 0.35,
    "I": 0.0,
    "Irrelevant": 0.0,
}



def keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?:s)?(?!\w)", re.IGNORECASE)


GIFT_CATEGORY_PATTERNS = {
    category: tuple(keyword_pattern(keyword) for keyword in keywords)
    for category, keywords in GIFT_CATEGORY_KEYWORDS.items()
}


def clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def make_product_text(row: dict, max_chars: int) -> str:
    fields = [
        ("Title", row.get("product_title")),
        ("Description", row.get("product_description")),
        ("Features", row.get("product_bullet_point")),
        ("Brand", row.get("product_brand")),
        ("Color", row.get("product_color")),
    ]
    text = "\n".join(
        f"{name}: {clean(value)}" for name, value in fields if clean(value)
    )
    return text[:max_chars]


def keep_english_task_one(row: dict) -> bool:
    return row.get("product_locale") == "us" and int(row.get("small_version", 0)) == 1


def get_gift_category(query: object) -> str | None:
    query_text = clean(query)
    for category, patterns in GIFT_CATEGORY_PATTERNS.items():
        if any(pattern.search(query_text) for pattern in patterns):
            return category
    return None


def keep_gift_query(row: dict) -> bool:
    return get_gift_category(row.get("query")) is not None


def sample_complete_queries(dataset: Dataset, max_pairs: int, seed: int) -> Dataset:
    """Select complete query groups until approximately max_pairs are included."""
    if max_pairs <= 0 or len(dataset) <= max_pairs:
        return dataset

    query_counts = Counter(dataset["query_id"])
    query_ids = list(query_counts)
    random.Random(seed).shuffle(query_ids)

    selected: set[int] = set()
    selected_pairs = 0
    for query_id in query_ids:
        count = query_counts[query_id]
        if selected and selected_pairs + count > max_pairs:
            continue
        selected.add(query_id)
        selected_pairs += count
        if selected_pairs >= max_pairs:
            break

    return dataset.filter(lambda row: row["query_id"] in selected)


def transform_split(
    dataset: Dataset,
    max_pairs: int,
    max_chars: int,
    seed: int,
    scope: str,
) -> Dataset:
    dataset = dataset.filter(keep_english_task_one)
    if scope == "gift":
        # Filter by query, not product text, so every candidate and negative for a
        # selected gift query remains in the same ranking group.
        dataset = dataset.filter(keep_gift_query)
    dataset = sample_complete_queries(dataset, max_pairs=max_pairs, seed=seed)

    def transform(row: dict) -> dict:
        label_name = str(row["esci_label"])
        if label_name not in LABEL_SCORES:
            raise ValueError(f"Unknown ESCI label: {label_name}")
        return {
            "sentence1": clean(row["query"]),
            "sentence2": make_product_text(row, max_chars=max_chars),
            "label": LABEL_SCORES[label_name],
            "query_id": int(row["query_id"]),
            "product_id": str(row["product_id"]),
            "esci_label": label_name,
            "gift_category": get_gift_category(row["query"]) or "general",
        }

    return dataset.map(transform, remove_columns=dataset.column_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-pairs", type=int, default=100_000)
    parser.add_argument("--validation-pairs", type=int, default=10_000)
    parser.add_argument("--max-product-chars", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scope",
        choices=("gift", "all"),
        default="gift",
        help="Prepare the 10 gift categories by default, or use all products.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source = load_dataset(args.dataset_id)
    if "train" not in source or "test" not in source:
        raise ValueError("The ESCI dataset must contain train and test splits.")

    prepared = DatasetDict(
        {
            "train": transform_split(
                source["train"],
                max_pairs=args.train_pairs,
                max_chars=args.max_product_chars,
                seed=args.seed,
                scope=args.scope,
            ),
            "validation": transform_split(
                source["test"],
                max_pairs=args.validation_pairs,
                max_chars=args.max_product_chars,
                seed=args.seed + 1,
                scope=args.scope,
            ),
        }
    )
    prepared.save_to_disk(str(output_dir))

    print(f"Saved dataset to {output_dir.resolve()}")
    print(f"Train pairs: {len(prepared['train']):,}")
    print(f"Validation pairs: {len(prepared['validation']):,}")
    print(f"Scope: {args.scope}")
    print(f"Train gift categories: {dict(Counter(prepared['train']['gift_category']))}")


if __name__ == "__main__":
    main()
