#!/usr/bin/env python3
"""Evaluate Amazon 1-to-5 rating prediction outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


VALID_RATINGS = {1, 2, 3, 4, 5}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_gold(value: Any) -> int:
    try:
        rating = int(round(float(value)))
    except Exception as exc:
        raise ValueError(f"Could not parse gold rating from {value!r}") from exc
    if rating not in VALID_RATINGS:
        raise ValueError(f"Gold rating must be in 1..5, got {value!r}")
    return rating


def parse_prediction(value: Any) -> int | None:
    for match in re.findall(r"\b[1-5]\b", str(value)):
        rating = int(match)
        if rating in VALID_RATINGS:
            return rating
    return None


def manual_metrics(y_true: list[int], y_pred: list[int], invalid_count: int, total: int) -> dict[str, Any]:
    print("Warning: sklearn is not available; macro_f1 and confusion_matrix are computed manually.")
    errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
    squared_errors = [(t - p) ** 2 for t, p in zip(y_true, y_pred)]
    per_rating = {}
    for rating in sorted(VALID_RATINGS):
        indices = [idx for idx, value in enumerate(y_true) if value == rating]
        per_rating[str(rating)] = (
            sum(1 for idx in indices if y_pred[idx] == rating) / len(indices) if indices else None
        )
    f1_scores = []
    for rating in sorted(VALID_RATINGS):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == rating and p == rating)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != rating and p == rating)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == rating and p != rating)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    confusion = [[0 for _ in VALID_RATINGS] for _ in VALID_RATINGS]
    for true, pred in zip(y_true, y_pred):
        confusion[true - 1][pred - 1] += 1
    return {
        "num_samples": total,
        "num_valid_predictions": len(y_pred),
        "invalid_prediction_count": invalid_count,
        "MAE": sum(errors) / len(errors) if errors else None,
        "RMSE": math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else None,
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_pred) if y_pred else None,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "per_rating_accuracy": per_rating,
        "confusion_matrix": confusion,
    }


def compute_metrics(gold: list[int], pred: list[int | None]) -> dict[str, Any]:
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if p is not None]
    y_true = [g for g, _ in valid_pairs]
    y_pred = [int(p) for _, p in valid_pairs if p is not None]
    invalid_count = len(pred) - len(y_pred)

    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error

        per_rating = {}
        by_rating: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for true, prediction in zip(y_true, y_pred):
            by_rating[true].append((true, prediction))
        for rating in sorted(VALID_RATINGS):
            pairs = by_rating.get(rating, [])
            per_rating[str(rating)] = (
                sum(1 for true, prediction in pairs if true == prediction) / len(pairs) if pairs else None
            )
        return {
            "num_samples": len(gold),
            "num_valid_predictions": len(y_pred),
            "invalid_prediction_count": invalid_count,
            "MAE": mean_absolute_error(y_true, y_pred) if y_pred else None,
            "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)) if y_pred else None,
            "accuracy": accuracy_score(y_true, y_pred) if y_pred else None,
            "macro_f1": f1_score(y_true, y_pred, labels=sorted(VALID_RATINGS), average="macro", zero_division=0) if y_pred else None,
            "per_rating_accuracy": per_rating,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(VALID_RATINGS)).tolist() if y_pred else [],
        }
    except ImportError:
        return manual_metrics(y_true, y_pred, invalid_count, len(gold))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Amazon 1-to-5 rating predictions.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--gold_col", default="label")
    parser.add_argument("--pred_col", default="prediction")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_path))
    if not rows:
        raise ValueError(f"No rows found in {args.input_path}")
    gold = [parse_gold(row[args.gold_col]) for row in rows]
    pred = [parse_prediction(row.get(args.pred_col, row.get("pred", ""))) for row in rows]
    metrics = compute_metrics(gold, pred)
    output = json.dumps(metrics, indent=2)
    print(output)
    if args.output_path:
        Path(args.output_path).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
