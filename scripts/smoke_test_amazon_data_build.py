#!/usr/bin/env python3
"""Standalone smoke test for the Amazon GraphCheck data builder."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="amazon_graphcheck_smoke_"))
    try:
        dataset_name = "Amazon_Test_Rating"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_amazon_graphcheck_data.py"),
                "--reviews_path",
                str(ROOT / "tests" / "fixtures" / "amazon_reviews_small.jsonl"),
                "--metadata_path",
                str(ROOT / "tests" / "fixtures" / "amazon_meta_small.jsonl"),
                "--output_dir",
                str(tmp_root),
                "--dataset_name",
                dataset_name,
                "--min_user_interactions",
                "5",
                "--history_k",
                "5",
            ],
            check=True,
        )

        dataset_root = tmp_root / dataset_name
        pkl_path = dataset_root / f"{dataset_name}.pkl"
        assert pkl_path.exists(), "pkl was not created"
        df = pd.read_pickle(pkl_path)
        assert len(df) > 0, "no samples were created"
        for column in ["doc_text", "claim_text", "doc_kg", "claim_kg", "label"]:
            assert column in df.columns, f"missing column {column}"
        assert set(df["label"].unique()).issubset({1, 2, 3, 4, 5}), "labels must be 1 to 5 ratings"
        assert set(df["task_type"].unique()) == {"rating"}, "task_type must be rating"
        assert 3 in set(df["label"].unique()), "3-star ratings must be retained"

        target_review_by_item = {
            "B006": "FUTURE_REVIEW_U1_B006",
            "B007": "FUTURE_REVIEW_U1_B007",
            "C006": "FUTURE_REVIEW_U2_C006",
            "D006": "FUTURE_REVIEW_U3_D006",
        }
        for _, row in df.iterrows():
            target_review = target_review_by_item.get(row["target_item_id"])
            if target_review:
                assert target_review not in row["doc_text"]
                assert target_review not in row["claim_text"]
            for triples in (row["doc_kg"], row["claim_kg"]):
                assert triples, "KG cannot be empty"
                for triple in triples:
                    assert len(triple) == 3
                    assert all(isinstance(part, str) and part.strip() for part in triple)

        for split_name in ["train_indices.txt", "val_indices.txt", "test_indices.txt"]:
            assert (dataset_root / "split" / split_name).exists(), f"missing {split_name}"

        print(f"Amazon data build smoke test passed with {len(df)} samples.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
