"""Fine-tune MiniLM CrossEncoder and optionally publish it to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from huggingface_hub import HfApi, login
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss


BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/genieai-product-reranker")
    parser.add_argument("--hub-repo", help="Example: username/genieai-product-reranker")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_from_disk(args.data_dir)
    train_data = data["train"].select_columns(["sentence1", "sentence2", "label"])
    validation_data = data["validation"].select_columns(
        ["sentence1", "sentence2", "label"]
    )

    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=args.max_length)
    use_fp16 = torch.cuda.is_available()

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    model.save_pretrained(str(final_dir))
    print(f"Saved final model to {final_dir.resolve()}")

    if args.hub_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN before using --hub-repo.")
        login(token=token)
        HfApi(token=token).create_repo(
            repo_id=args.hub_repo,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        model.push_to_hub(args.hub_repo, private=True, token=token)
        print(f"Published model to https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()

