#!/usr/bin/env python3
"""Run an end-to-end Amazon user-simulation smoke pipeline."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = [
    "doc_text",
    "claim_text",
    "doc_kg",
    "claim_kg",
    "label",
    "user_id",
    "target_item_id",
    "target_rating",
    "target_timestamp",
    "history_item_ids",
    "task_type",
]


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def item_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("parent_asin") or record.get("asin") or "").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_reviews_from_hf(args: argparse.Namespace, review_out: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if args.reviews_path:
        rows = read_jsonl(Path(args.reviews_path))
        if args.force_rebuild or not review_out.exists():
            write_jsonl(review_out, rows)
        return rows, {item_id_from_record(row) for row in rows if item_id_from_record(row)}

    if review_out.exists() and not args.force_rebuild:
        print(f"Reusing existing review sample: {review_out}")
        rows = read_jsonl(review_out)
        return rows, {item_id_from_record(row) for row in rows if item_id_from_record(row)}

    if args.skip_hf_sample:
        raise FileNotFoundError(f"--skip_hf_sample was set but review sample does not exist: {review_out}")

    from datasets import load_dataset

    print(f"Loading review stream: raw_review_{args.category}")
    review_stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{args.category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(review_stream):
        if idx >= args.max_scan_reviews:
            break
        row = dict(row)
        user_id = row.get("user_id")
        item_id = item_id_from_record(row)
        if not user_id or not item_id or row.get("rating") is None or row.get("timestamp") is None:
            continue
        by_user[str(user_id)].append(row)
        dense_users = [user for user, rows in by_user.items() if len(rows) >= args.min_interactions]
        if len(dense_users) >= args.target_users:
            break

    selected_users = [user for user, rows in by_user.items() if len(rows) >= args.min_interactions][: args.target_users]
    selected_reviews = []
    needed_items: set[str] = set()
    for user_id in selected_users:
        rows = sorted(by_user[user_id], key=lambda row: row.get("timestamp", 0))
        selected_reviews.extend(rows)
        needed_items.update(item_id_from_record(row) for row in rows if item_id_from_record(row))

    if not selected_reviews:
        raise RuntimeError("No dense users found. Increase --max_scan_reviews or lower --min_interactions.")

    write_jsonl(review_out, selected_reviews)
    print(f"Selected users: {len(selected_users)}")
    print(f"Selected reviews: {len(selected_reviews)}")
    print(f"Wrote reviews: {review_out}")
    return selected_reviews, needed_items


def filter_metadata_stream(rows: Any, needed_items: set[str], max_scan_meta: int) -> tuple[list[dict[str, Any]], set[str]]:
    selected_meta = []
    found_items: set[str] = set()
    for idx, row in enumerate(rows):
        if idx >= max_scan_meta:
            break
        row = dict(row)
        item_id = item_id_from_record(row)
        if item_id in needed_items and item_id not in found_items:
            selected_meta.append(row)
            found_items.add(item_id)
        if found_items >= needed_items:
            break
    return selected_meta, found_items


def stream_raw_metadata(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid metadata JSON at {path}:{line_no}: {exc}")


def download_raw_metadata(category: str, raw_output_dir: Path) -> Path:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_output_dir / f"meta_{category}.jsonl.gz"
    if raw_path.exists():
        print(f"Reusing raw metadata file: {raw_path}")
        return raw_path
    url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"
    print(f"Downloading raw metadata: {url}")
    urllib.request.urlretrieve(url, raw_path)
    return raw_path


def sample_metadata(args: argparse.Namespace, meta_out: Path, needed_items: set[str]) -> tuple[int, int]:
    if args.metadata_path:
        rows = read_jsonl(Path(args.metadata_path))
        if args.force_rebuild or not meta_out.exists():
            write_jsonl(meta_out, rows)
        found_items = {item_id_from_record(row) for row in rows if item_id_from_record(row) in needed_items}
        print_metadata_report(needed_items, found_items, meta_out)
        return len(needed_items), len(found_items)

    if meta_out.exists() and not args.force_rebuild:
        print(f"Reusing existing metadata sample: {meta_out}")
        rows = read_jsonl(meta_out)
        found_items = {item_id_from_record(row) for row in rows if item_id_from_record(row) in needed_items}
        print_metadata_report(needed_items, found_items, meta_out)
        return len(needed_items), len(found_items)

    if args.skip_hf_sample:
        raise FileNotFoundError(f"--skip_hf_sample was set but metadata sample does not exist: {meta_out}")

    selected_meta: list[dict[str, Any]] = []
    found_items: set[str] = set()
    try:
        from datasets import load_dataset

        print(f"Loading metadata stream: raw_meta_{args.category}")
        meta_stream = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{args.category}",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
        selected_meta, found_items = filter_metadata_stream(meta_stream, needed_items, args.max_scan_meta)
    except Exception as exc:
        print(f"Warning: HuggingFace metadata streaming failed: {exc}")
        raw_path = download_raw_metadata(args.category, Path(args.raw_output_dir))
        selected_meta, found_items = filter_metadata_stream(stream_raw_metadata(raw_path), needed_items, args.max_scan_meta)

    write_jsonl(meta_out, selected_meta)
    print_metadata_report(needed_items, found_items, meta_out)
    return len(needed_items), len(found_items)


def print_metadata_report(needed_items: set[str], found_items: set[str], meta_out: Path) -> None:
    missing = needed_items - found_items
    print(f"Needed metadata items: {len(needed_items)}")
    print(f"Found metadata items: {len(found_items)}")
    print(f"Missing metadata count: {len(missing)}")
    print(f"Metadata output path: {meta_out}")
    if missing:
        print("Warning: some metadata was not found. Fallback triples will be used where needed.")
        print(f"First missing item ids: {sorted(missing)[:10]}")


def run_build_data(args: argparse.Namespace, review_out: Path, meta_out: Path) -> Path:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_amazon_graphcheck_data.py"),
        "--reviews_path",
        str(review_out),
        "--metadata_path",
        str(meta_out),
        "--output_dir",
        str(args.output_dir),
        "--dataset_name",
        args.dataset_name,
        "--min_user_interactions",
        str(args.min_user_interactions),
        "--history_k",
        str(args.history_k),
        "--positive_threshold",
        str(args.positive_threshold),
        "--negative_threshold",
        str(args.negative_threshold),
        "--drop_neutral",
        str(args.drop_neutral).lower(),
        "--verified_only",
        str(args.verified_only).lower(),
        "--split_strategy",
        args.split_strategy,
        "--max_history_text_chars",
        str(args.max_history_text_chars),
        "--max_item_text_chars",
        str(args.max_item_text_chars),
        "--max_features_per_item",
        str(args.max_features_per_item),
        "--max_aspects_per_user",
        str(args.max_aspects_per_user),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    dataset_root = Path(args.output_dir) / args.dataset_name
    pkl_path = dataset_root / f"{args.dataset_name}.pkl"
    required_paths = [
        pkl_path,
        dataset_root / "split" / "train_indices.txt",
        dataset_root / "split" / "val_indices.txt",
        dataset_root / "split" / "test_indices.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Amazon data build did not create required files: {missing}")
    return pkl_path


def validate_triples(triples: Any, field: str, row_idx: int) -> None:
    if not isinstance(triples, list) or not triples:
        raise ValueError(f"Row {row_idx} {field} must be a non-empty list")
    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError(f"Row {row_idx} {field} contains invalid triple: {triple!r}")
        if not all(isinstance(part, str) and part.strip() for part in triple):
            raise ValueError(f"Row {row_idx} {field} contains empty triple element: {triple!r}")


def normalize_for_leakage_check(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def iter_doc_item_nodes(triples: Any) -> set[str]:
    item_nodes = set()
    if not isinstance(triples, list):
        return item_nodes
    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        for node in (triple[0], triple[2]):
            node_text = str(node or "")
            if node_text.startswith("item:"):
                item_nodes.add(normalize_for_leakage_check(node_text.removeprefix("item:")))
    return item_nodes


def validate_no_target_leakage(df: pd.DataFrame, review_rows: list[dict[str, Any]]) -> dict[str, int]:
    review_lookup = {}
    for row in review_rows:
        key = (str(row.get("user_id")), item_id_from_record(row), int(row.get("timestamp", 0)))
        review_lookup[key] = [str(row.get("title") or ""), str(row.get("text") or "")]

    skipped_short_title = 0
    skipped_short_text = 0
    for idx, row in df.iterrows():
        target_item_id = str(row["target_item_id"])
        target_item_norm = normalize_for_leakage_check(target_item_id)
        history_item_ids = row.get("history_item_ids", [])
        if not isinstance(history_item_ids, list):
            history_item_ids = []
        history_item_norms = {normalize_for_leakage_check(item_id) for item_id in history_item_ids}
        if target_item_norm in history_item_norms:
            raise ValueError(f"Target item leakage detected in history_item_ids for sample {idx}: {target_item_id!r}")

        if target_item_norm in iter_doc_item_nodes(row["doc_kg"]):
            raise ValueError(f"Target item leakage detected in doc_kg item nodes for sample {idx}: {target_item_id!r}")

        key = (str(row["user_id"]), str(row["target_item_id"]), int(row["target_timestamp"]))
        for field_name, value in zip(("title", "text"), review_lookup.get(key, [])):
            normalized_value = normalize_for_leakage_check(value)
            if not normalized_value:
                continue
            if len(normalized_value) < 50:
                if field_name == "title":
                    skipped_short_title += 1
                else:
                    skipped_short_text += 1
                continue
            haystacks = {
                "doc_text": row["doc_text"],
                "claim_text": row["claim_text"],
                "doc_kg": json.dumps(row["doc_kg"], ensure_ascii=False),
                "claim_kg": json.dumps(row["claim_kg"], ensure_ascii=False),
            }
            for haystack_name, haystack in haystacks.items():
                if normalized_value in normalize_for_leakage_check(haystack):
                    raise ValueError(
                        f"Target review leakage detected in sample {idx} {haystack_name}: {str(value)[:80]!r}"
                    )
    return {
        "skipped_short_target_titles": skipped_short_title,
        "skipped_short_target_texts": skipped_short_text,
    }


def validate_pkl(pkl_path: Path, review_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_pickle(pkl_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"PKL is missing required columns: {missing}")
    labels = [int(label) for label in df["label"]]
    if not set(labels).issubset({1, 2, 3, 4, 5}):
        raise ValueError(f"Labels must be ratings in {{1, 2, 3, 4, 5}}, got {sorted(set(labels))}")
    if set(df["task_type"].astype(str)) != {"rating"}:
        raise ValueError(f"Amazon task_type must be 'rating', got {sorted(set(df['task_type'].astype(str)))}")
    rating_three_rows = [
        idx for idx, row in df.iterrows()
        if int(round(float(row["target_rating"]))) == 3
    ]
    if rating_three_rows and not all(int(df.loc[idx, "label"]) == 3 for idx in rating_three_rows):
        raise ValueError("Rows with target rating 3 must be retained with label 3.")
    for idx, row in df.iterrows():
        validate_triples(row["doc_kg"], "doc_kg", idx)
        validate_triples(row["claim_kg"], "claim_kg", idx)
    leakage_report = validate_no_target_leakage(df, review_rows)

    rating_counts = Counter(df["target_rating"])
    label_counts = Counter(labels)
    print("\nPKL validation")
    print(f"Num samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rating distribution: {dict(label_counts)}")
    print(f"Target rating distribution: {dict(rating_counts)}")
    print(f"Unique rating labels: {len(label_counts)}")
    print(f"Skipped short target titles in leakage check: {leakage_report['skipped_short_target_titles']}")
    print(f"Skipped short target texts in leakage check: {leakage_report['skipped_short_target_texts']}")
    if len(label_counts) < 3:
        print("Warning: fewer than 3 unique ratings found. This is suitable for smoke testing but weak for evaluation.")
    if label_counts and max(label_counts.values()) / len(df) > 0.8:
        print("Warning: one rating accounts for more than 80% of samples.")
    print("First sample preview:")
    print(str(df.iloc[0][["user_id", "target_item_id", "label", "doc_text", "claim_text"]])[:1200])
    print(f"First doc_kg triples: {df.iloc[0]['doc_kg'][:5]}")
    print(f"First claim_kg triples: {df.iloc[0]['claim_kg'][:5]}")
    return df, {"rating_distribution": dict(label_counts), "unique_rating_labels": len(label_counts)}


def run_graph_build(args: argparse.Namespace) -> None:
    if args.skip_graph_build:
        print("Skipping graph_build.py")
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "graph_build.py"),
        "--data_name",
        args.dataset_name,
        "--kg_root",
        str(Path(args.output_dir).resolve()),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def validate_one_graph(path: Path, label: str) -> dict[str, tuple[int, ...]]:
    graph = torch.load(path, map_location="cpu")
    for attr in ("x", "edge_index", "edge_attr"):
        if not hasattr(graph, attr):
            raise ValueError(f"{label} graph is missing {attr}: {path}")
    if graph.x.ndim != 2:
        raise ValueError(f"{label} graph x must be 2D, got {graph.x.shape}")
    if graph.edge_index.shape[0] != 2:
        raise ValueError(f"{label} edge_index first dimension must be 2, got {graph.edge_index.shape}")
    if graph.edge_attr.ndim != 2:
        raise ValueError(f"{label} edge_attr must be 2D, got {graph.edge_attr.shape}")
    if graph.num_nodes <= 0:
        raise ValueError(f"{label} graph must have at least one node")
    if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
        raise ValueError(f"{label} edge count and edge_attr rows differ")
    return {
        "x": tuple(graph.x.shape),
        "edge_index": tuple(graph.edge_index.shape),
        "edge_attr": tuple(graph.edge_attr.shape),
    }


def validate_graph_files(output_dir: Path, dataset_name: str, num_samples: int) -> tuple[int, int]:
    dataset_root = output_dir / dataset_name
    doc_dir = dataset_root / "graphs" / "doc"
    claim_dir = dataset_root / "graphs" / "claim"
    if not doc_dir.exists() or not claim_dir.exists():
        raise FileNotFoundError(f"Graph directories are missing under {dataset_root}")
    doc_graphs = sorted(doc_dir.glob("*.pt"))
    claim_graphs = sorted(claim_dir.glob("*.pt"))
    if len(doc_graphs) != num_samples:
        raise ValueError(f"Doc graph count {len(doc_graphs)} != sample count {num_samples}")
    if len(claim_graphs) != num_samples:
        raise ValueError(f"Claim graph count {len(claim_graphs)} != sample count {num_samples}")
    doc_shapes = validate_one_graph(doc_graphs[0], "doc")
    claim_shapes = validate_one_graph(claim_graphs[0], "claim")
    print("\nGraph validation")
    print(f"Doc graph count: {len(doc_graphs)}")
    print(f"Claim graph count: {len(claim_graphs)}")
    print(f"Doc x shape: {doc_shapes['x']}")
    print(f"Doc edge_index shape: {doc_shapes['edge_index']}")
    print(f"Doc edge_attr shape: {doc_shapes['edge_attr']}")
    print(f"Claim x shape: {claim_shapes['x']}")
    print(f"Claim edge_index shape: {claim_shapes['edge_index']}")
    print(f"Claim edge_attr shape: {claim_shapes['edge_attr']}")
    return len(doc_graphs), len(claim_graphs)


def validate_dataset_loading(output_dir: Path, dataset_name: str, num_samples: int) -> int:
    os.environ["GRAPHCHECK_KG_ROOT"] = str(output_dir.resolve())
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from dataset.utils.collate import collate_fn
    from dataset.utils.dataset import KGDataset

    ds = KGDataset(dataset_name)
    if len(ds) != num_samples:
        raise ValueError(f"KGDataset length {len(ds)} != sample count {num_samples}")
    sample = ds[0]
    required_keys = {"id", "label", "claim_kg", "doc_kg", "claim_kg_text", "doc_kg_text", "text", "index", "dataset"}
    missing = required_keys - set(sample)
    if missing:
        raise ValueError(f"KGDataset sample is missing keys: {missing}")
    if sample["label"] not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Invalid sample label: {sample['label']}")
    if dataset_name.startswith("Amazon"):
        if "predict the rating" not in sample["text"]:
            raise ValueError("Amazon rating prompt was not used by KGDataset")
        if "support" in sample["text"].lower() or "unsupport" in sample["text"].lower():
            raise ValueError("Amazon prompt must not contain support/unsupport")
    batch = collate_fn([sample])
    if "text" not in batch or "label" not in batch or len(batch["text"]) != 1:
        raise ValueError("collate_fn failed to batch one sample")

    print("\nKGDataset validation")
    print(f"Dataset length: {len(ds)}")
    print(f"Sample label: {sample['label']}")
    print(f"Prompt preview: {sample['text'][:500]}")
    print(f"Batch keys: {list(batch.keys())}")
    print("Collate test: passed")
    return len(ds)


def run_train_smoke(args: argparse.Namespace) -> str:
    if not args.run_train_smoke:
        print("Train smoke: skipped")
        return "skipped"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--project",
        args.project,
        "--train_dataset",
        args.dataset_name,
        "--llm_model_name",
        args.llm_model_name,
        "--gnn_model_name",
        args.gnn_model_name,
        "--batch_size",
        "1",
        "--eval_batch_size",
        "1",
        "--grad_steps",
        "1",
        "--num_epochs",
        "1",
        "--patience",
        "1",
        "--max_txt_len",
        "512",
        "--max_new_tokens",
        "5",
        "--lr",
        "1e-5",
    ]
    env = os.environ.copy()
    env["GRAPHCHECK_KG_ROOT"] = str(Path(args.output_dir).resolve())
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    print("Train smoke: passed")
    return "passed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Amazon GraphCheck smoke pipeline.")
    parser.add_argument("--category", default="CDs_and_Vinyl")
    parser.add_argument("--dataset_name", default="Amazon_CDs_HF_Rating_Smoke")
    parser.add_argument("--output_dir", default="dataset/extracted_KG")
    parser.add_argument("--hf_output_dir", default="data/amazon2023/hf_smoke")
    parser.add_argument("--raw_output_dir", default="data/amazon2023/raw")
    parser.add_argument("--target_users", type=int, default=20)
    parser.add_argument("--min_interactions", type=int, default=3)
    parser.add_argument("--max_scan_reviews", type=int, default=200000)
    parser.add_argument("--max_scan_meta", type=int, default=1000000)
    parser.add_argument("--min_user_interactions", type=int, default=2)
    parser.add_argument("--history_k", type=int, default=3)
    parser.add_argument("--positive_threshold", type=int, default=4)
    parser.add_argument("--negative_threshold", type=int, default=2)
    parser.add_argument("--drop_neutral", type=str_to_bool, default=True)
    parser.add_argument("--verified_only", type=str_to_bool, default=False)
    parser.add_argument("--split_strategy", default="temporal_global", choices=["temporal_global", "temporal_user"])
    parser.add_argument("--max_history_text_chars", type=int, default=3000)
    parser.add_argument("--max_item_text_chars", type=int, default=1200)
    parser.add_argument("--max_features_per_item", type=int, default=10)
    parser.add_argument("--max_aspects_per_user", type=int, default=20)
    parser.add_argument("--skip_hf_sample", action="store_true")
    parser.add_argument("--skip_graph_build", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--run_dataset_check", type=str_to_bool, default=True)
    parser.add_argument("--run_train_smoke", action="store_true")
    parser.add_argument("--llm_model_name", default="qwen_0_5b")
    parser.add_argument("--gnn_model_name", default="gt")
    parser.add_argument("--project", default="Amazon_Rating_Smoke_Test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_type", default="rating", help="Compatibility argument; only 'rating' is supported")
    parser.add_argument("--reviews_path", default=None, help="Optional local review JSONL for fixture smoke tests")
    parser.add_argument("--metadata_path", default=None, help="Optional local metadata JSONL for fixture smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_type != "rating":
        raise ValueError("Amazon smoke pipeline only supports task_type=rating.")
    args.output_dir = str((REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir))
    args.hf_output_dir = str((REPO_ROOT / args.hf_output_dir).resolve() if not Path(args.hf_output_dir).is_absolute() else Path(args.hf_output_dir))
    args.raw_output_dir = str((REPO_ROOT / args.raw_output_dir).resolve() if not Path(args.raw_output_dir).is_absolute() else Path(args.raw_output_dir))

    hf_output_dir = Path(args.hf_output_dir)
    review_out = hf_output_dir / f"{args.category}_reviews_sample.jsonl"
    meta_out = hf_output_dir / f"{args.category}_meta_sample.jsonl"

    review_rows, needed_items = sample_reviews_from_hf(args, review_out)
    sample_metadata(args, meta_out, needed_items)
    pkl_path = run_build_data(args, review_out, meta_out)
    df, pkl_report = validate_pkl(pkl_path, review_rows)

    graph_counts = (0, 0)
    dataset_length = 0
    if not args.skip_graph_build:
        run_graph_build(args)
        graph_counts = validate_graph_files(Path(args.output_dir), args.dataset_name, len(df))
        if args.run_dataset_check:
            dataset_length = validate_dataset_loading(Path(args.output_dir), args.dataset_name, len(df))
    elif args.run_dataset_check:
        print("Skipping KGDataset validation because graph_build.py was skipped.")

    train_status = run_train_smoke(args)

    print("\nAmazon rating smoke pipeline completed.")
    print("Task: 1 to 5 rating prediction")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Review sample path: {review_out}")
    print(f"Metadata sample path: {meta_out}")
    print(f"PKL path: {pkl_path}")
    print(f"Number of samples: {len(df)}")
    print(f"Rating distribution: {pkl_report['rating_distribution']}")
    print(f"Unique rating labels: {pkl_report['unique_rating_labels']}")
    print(f"Doc graph count: {graph_counts[0]}")
    print(f"Claim graph count: {graph_counts[1]}")
    print(f"KGDataset length: {dataset_length}")
    print(f"Collate test: {'passed' if dataset_length else 'skipped'}")
    print(f"Train smoke: {train_status}")


if __name__ == "__main__":
    main()
