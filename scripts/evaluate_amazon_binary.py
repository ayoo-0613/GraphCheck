#!/usr/bin/env python3
"""Evaluate Amazon binary support/unsupport outputs."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


LABEL_MAP = {"support": 1, "unsupport": 0, "1": 1, "0": 0}


def parse_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    matches = re.findall(r"unsupport|support", text, re.IGNORECASE)
    if matches:
        return LABEL_MAP[matches[0].lower()]
    raise ValueError(f"Could not parse label from {value!r}")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }
    except ImportError:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        total = len(y_true)
        accuracy = (tp + tn) / total if total else 0.0
        recall_pos = tp / (tp + fn) if tp + fn else 0.0
        recall_neg = tn / (tn + fp) if tn + fp else 0.0
        precision_pos = tp / (tp + fp) if tp + fp else 0.0
        precision_neg = tn / (tn + fn) if tn + fn else 0.0
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if precision_pos + recall_pos else 0.0
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg else 0.0
        return {
            "accuracy": accuracy,
            "balanced_accuracy": (recall_pos + recall_neg) / 2,
            "macro_f1": (f1_pos + f1_neg) / 2,
            "precision": (precision_pos + precision_neg) / 2,
            "recall": (recall_pos + recall_neg) / 2,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Amazon binary GraphCheck outputs.")
    parser.add_argument("--input_path", required=True, help="JSONL output file with label and pred fields.")
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    if not rows:
        raise ValueError(f"No rows found in {args.input_path}")
    y_true = [parse_label(row["label"]) for row in rows]
    y_pred = [parse_label(row.get("pred", row.get("prediction", ""))) for row in rows]
    print(json.dumps(compute_metrics(y_true, y_pred), indent=2))


if __name__ == "__main__":
    main()
