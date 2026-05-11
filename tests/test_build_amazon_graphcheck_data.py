import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_build_amazon_graphcheck_data(tmp_path):
    reviews_path = ROOT / "tests" / "fixtures" / "amazon_reviews_small.jsonl"
    metadata_path = ROOT / "tests" / "fixtures" / "amazon_meta_small.jsonl"
    dataset_name = "Amazon_Test_Rating"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_amazon_graphcheck_data.py"),
            "--reviews_path",
            str(reviews_path),
            "--metadata_path",
            str(metadata_path),
            "--output_dir",
            str(tmp_path),
            "--dataset_name",
            dataset_name,
            "--min_user_interactions",
            "5",
            "--history_k",
            "5",
        ],
        check=True,
    )

    dataset_root = tmp_path / dataset_name
    pkl_path = dataset_root / f"{dataset_name}.pkl"
    assert pkl_path.exists()

    df = pd.read_pickle(pkl_path)
    assert len(df) > 0
    for column in ["doc_text", "claim_text", "doc_kg", "claim_kg", "label"]:
        assert column in df.columns

    target_review_by_item = {
        "B006": "FUTURE_REVIEW_U1_B006",
        "B007": "FUTURE_REVIEW_U1_B007",
        "C006": "FUTURE_REVIEW_U2_C006",
        "D006": "FUTURE_REVIEW_U3_D006",
    }
    target_title_by_item = {
        "B006": "FUTURE_TITLE_U1_B006",
        "B007": "FUTURE_TITLE_U1_B007",
        "C006": "FUTURE_TITLE_U2_C006",
        "D006": "FUTURE_TITLE_U3_D006",
    }
    for _, row in df.iterrows():
        target_review = target_review_by_item.get(row["target_item_id"])
        if target_review:
            assert target_review not in row["doc_text"]
            assert target_review not in row["claim_text"]
        target_title = target_title_by_item.get(row["target_item_id"])
        if target_title:
            assert target_title not in row["doc_text"]
            assert target_title not in row["claim_text"]

    for triples in list(df["doc_kg"]) + list(df["claim_kg"]):
        assert triples
        for triple in triples:
            assert len(triple) == 3
            assert all(isinstance(part, str) and part.strip() for part in triple)

    assert set(df["label"].unique()).issubset({1, 2, 3, 4, 5})
    assert set(df["task_type"].unique()) == {"rating"}
    assert 3 in set(df["label"].unique())
    assert 0 not in set(df["label"].unique())
    assert (dataset_root / "split" / "train_indices.txt").exists()
    assert (dataset_root / "split" / "val_indices.txt").exists()
    assert (dataset_root / "split" / "test_indices.txt").exists()
