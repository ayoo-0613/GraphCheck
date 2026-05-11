import importlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data.data import Data

from scripts.run_amazon_smoke_pipeline import validate_no_target_leakage


ROOT = Path(__file__).resolve().parents[1]


def write_minimal_graphs(dataset_root: Path, num_samples: int) -> None:
    for graph_kind in ["doc", "claim"]:
        graph_dir = dataset_root / "graphs" / graph_kind
        graph_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num_samples):
            data = Data(
                x=torch.ones((2, 1024)),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.ones((1, 1024)),
                num_nodes=2,
            )
            torch.save(data, graph_dir / f"{idx}.pt")


def test_amazon_smoke_pipeline_data_and_dataset(tmp_path, monkeypatch):
    dataset_name = "Amazon_Test_Rating"
    reviews_path = ROOT / "tests" / "fixtures" / "amazon_reviews_small.jsonl"
    metadata_path = ROOT / "tests" / "fixtures" / "amazon_meta_small.jsonl"

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
    for split_name in ["train_indices.txt", "val_indices.txt", "test_indices.txt"]:
        assert (dataset_root / "split" / split_name).exists()

    df = pd.read_pickle(pkl_path)
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({1, 2, 3, 4, 5})
    assert set(df["task_type"].unique()) == {"rating"}
    assert 3 in set(df["label"].unique())
    assert 0 not in set(df["label"].unique())
    assert "support" not in set(str(label).lower() for label in df["label"])
    assert "unsupport" not in set(str(label).lower() for label in df["label"])
    write_minimal_graphs(dataset_root, len(df))
    assert list((dataset_root / "graphs" / "doc").glob("*.pt"))
    assert list((dataset_root / "graphs" / "claim").glob("*.pt"))

    for _, row in df.iterrows():
        assert "FUTURE_REVIEW" not in row["doc_text"]
        assert "FUTURE_REVIEW" not in row["claim_text"]
        for triples in [row["doc_kg"], row["claim_kg"]]:
            assert triples
            for triple in triples:
                assert len(triple) == 3
                assert all(isinstance(part, str) and part.strip() for part in triple)

    monkeypatch.setenv("GRAPHCHECK_KG_ROOT", str(tmp_path))
    import dataset.utils.dataset as dataset_module

    importlib.reload(dataset_module)
    from dataset.utils.collate import collate_fn

    ds = dataset_module.KGDataset(dataset_name)
    assert len(ds) == len(df)
    sample = ds[0]
    assert sample["label"] in {"1", "2", "3", "4", "5"}
    assert "predict the rating" in sample["text"]
    assert "Please answer with one number only" in sample["text"]
    assert "1, 2, 3, 4, or 5" in sample["text"]
    assert "support" not in sample["text"].lower()
    assert "unsupport" not in sample["text"].lower()
    batch = collate_fn([sample])
    assert len(batch["text"]) == 1
    assert len(batch["label"]) == 1


def test_short_target_review_text_in_history_is_not_leakage():
    review_rows = [
        {
            "user_id": "u1",
            "parent_asin": "B0003QJRK0",
            "timestamp": 100,
            "title": "great cd find",
            "text": "Good cd",
        }
    ]
    df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "target_item_id": "B0003QJRK0",
                "target_timestamp": 100,
                "history_item_ids": ["B000OTHER"],
                "doc_text": "Historical review for another item: Good cd",
                "claim_text": "Candidate item metadata only",
                "doc_kg": [["user:current_user", "reviewed", "item:b000other"]],
                "claim_kg": [["item:target_item", "has_metadata", "metadata:available"]],
            }
        ]
    )

    report = validate_no_target_leakage(df, review_rows)

    assert report["skipped_short_target_titles"] == 1
    assert report["skipped_short_target_texts"] == 1


def test_long_unique_target_review_text_in_doc_text_is_leakage():
    review_rows = [
        {
            "user_id": "u1",
            "parent_asin": "B0003QJRK0",
            "timestamp": 100,
            "title": "ordinary title",
            "text": "This is a uniquely identifying target review sentence.",
        }
    ]
    df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "target_item_id": "B0003QJRK0",
                "target_timestamp": 100,
                "history_item_ids": ["B000OTHER"],
                "doc_text": "History accidentally includes this is a uniquely identifying target review sentence.",
                "claim_text": "Candidate item metadata only",
                "doc_kg": [["user:current_user", "reviewed", "item:b000other"]],
                "claim_kg": [["item:target_item", "has_metadata", "metadata:available"]],
            }
        ]
    )

    try:
        validate_no_target_leakage(df, review_rows)
    except ValueError as exc:
        assert "Target review leakage detected" in str(exc)
    else:
        raise AssertionError("Expected long target review leakage to fail validation")
