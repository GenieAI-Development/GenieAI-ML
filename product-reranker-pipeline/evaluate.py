"""Evaluate a local or Hub CrossEncoder on grouped ESCI validation queries."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sentence_transformers import CrossEncoder


def dcg(relevances: list[float], limit: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances[:limit])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/genieai-product-reranker/final")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/metrics.json")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = load_from_disk(args.data_dir)["validation"]
    model = CrossEncoder(args.model)

    pairs = list(zip(validation["sentence1"], validation["sentence2"]))
    scores = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for query_id, relevance, score in zip(
        validation["query_id"], validation["label"], scores
    ):
        grouped[int(query_id)].append((float(score), float(relevance)))

    ndcg_values: list[float] = []
    reciprocal_ranks: list[float] = []
    hits_at_four: list[float] = []

    for candidates in grouped.values():
        ranked = [item[1] for item in sorted(candidates, reverse=True)]
        ideal = sorted((item[1] for item in candidates), reverse=True)
        ideal_dcg = dcg(ideal, 10)
        if ideal_dcg > 0:
            ndcg_values.append(dcg(ranked, 10) / ideal_dcg)

        strong_positions = [
            position for position, relevance in enumerate(ranked, start=1) if relevance >= 0.7
        ]
        if any(relevance >= 0.7 for relevance in ideal):
            reciprocal_ranks.append(1 / strong_positions[0] if strong_positions else 0.0)
            hits_at_four.append(
                1.0 if strong_positions and strong_positions[0] <= 4 else 0.0
            )

    metrics = {
        "model": args.model,
        "queries": len(grouped),
        "pairs": len(validation),
        "ndcg_at_10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hit_rate_at_4": float(np.mean(hits_at_four)) if hits_at_four else 0.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

