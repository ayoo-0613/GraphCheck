#!/usr/bin/env python3
"""Build Amazon Reviews 2023 data in GraphCheck's dataframe format."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd


ASPECT_VOCAB = [
    "price",
    "quality",
    "delivery",
    "packaging",
    "size",
    "durability",
    "scent",
    "taste",
    "comfort",
    "ease_of_use",
    "battery",
    "design",
    "value",
    "performance",
    "fit",
]

POSITIVE_CUES = [
    "good",
    "great",
    "excellent",
    "love",
    "loved",
    "nice",
    "perfect",
    "useful",
    "easy",
    "comfortable",
    "high quality",
    "worth",
]

NEGATIVE_CUES = [
    "bad",
    "poor",
    "terrible",
    "hate",
    "disappointed",
    "expensive",
    "broken",
    "difficult",
    "uncomfortable",
    "low quality",
    "not worth",
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


def normalize_text_node(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_:+.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] or "unknown"


def safe_truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def price_to_range(price: Any) -> str:
    if price is None or price == "":
        return "unknown"
    if isinstance(price, (int, float)):
        value = float(price)
    else:
        match = re.search(r"\d+(?:\.\d+)?", str(price).replace(",", ""))
        if not match:
            return "unknown"
        value = float(match.group(0))
    if value < 10:
        return "low"
    if value < 50:
        return "medium"
    if value < 100:
        return "high"
    return "premium"


def timestamp_to_bucket(timestamp: Any) -> str:
    try:
        ts = int(float(timestamp))
        if ts > 10_000_000_000:
            ts = ts // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return f"{dt.year:04d}_{dt.month:02d}"
    except Exception:
        return "unknown"


def item_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("parent_asin") or record.get("asin") or "").strip()


def item_node(item_id: str, fallback: str = "unknown_item") -> str:
    return f"item:{normalize_text_node(item_id or fallback)}"


def typed_node(prefix: str, value: Any) -> str:
    return f"{prefix}:{normalize_text_node(str(value or 'unknown'))}"


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def extract_categories(metadata: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for key in ("main_category", "categories"):
        for value in as_list(metadata.get(key)):
            if isinstance(value, list):
                categories.extend(str(v) for v in value if v)
            elif value:
                categories.append(str(value))
    return dedupe_preserve_order(categories)


def extract_features(metadata: dict[str, Any]) -> list[str]:
    features: list[str] = []
    for value in as_list(metadata.get("features")):
        if isinstance(value, list):
            features.extend(str(v) for v in value if v)
        elif value:
            features.append(str(value))
    return dedupe_preserve_order(features)


def extract_description(metadata: dict[str, Any]) -> str:
    return " ".join(str(v) for v in as_list(metadata.get("description")) if v)


def selected_details(metadata: dict[str, Any], max_items: int = 8) -> list[str]:
    details = metadata.get("details")
    if not isinstance(details, dict):
        return []
    selected = []
    for key in sorted(details):
        value = details[key]
        if value:
            selected.append(f"{key}: {value}")
        if len(selected) >= max_items:
            break
    return selected


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        clean = re.sub(r"\s+", " ", str(value or "")).strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            output.append(clean)
    return output


def clean_review(record: dict[str, Any]) -> dict[str, Any]:
    item_id = item_id_from_record(record)
    if not record.get("user_id"):
        raise ValueError("Review row is missing required field user_id")
    if not item_id:
        raise ValueError("Review row is missing required field parent_asin or asin")
    if record.get("rating") is None:
        raise ValueError("Review row is missing required field rating")
    if record.get("timestamp") is None:
        raise ValueError("Review row is missing required field timestamp")
    cleaned = dict(record)
    cleaned["item_id"] = item_id
    cleaned["rating"] = float(record["rating"])
    cleaned["timestamp"] = int(float(record["timestamp"]))
    return cleaned


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file does not exist: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return rows


def build_metadata_map(metadata_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata_map = {}
    for row in metadata_rows:
        item_id = item_id_from_record(row)
        if item_id:
            metadata_map[item_id] = row
    return metadata_map


def build_history_text(
    history: list[dict[str, Any]],
    metadata_map: dict[str, dict[str, Any]],
    max_chars: int = 4000,
) -> str:
    lines = ["User historical behavior before the target interaction:"]
    for idx, interaction in enumerate(history, start=1):
        meta = metadata_map.get(interaction["item_id"], {})
        title = meta.get("title") or interaction["item_id"]
        category = (extract_categories(meta) or ["unknown"])[0]
        store = meta.get("store") or "unknown"
        review_text = safe_truncate(interaction.get("text", ""), 500)
        lines.extend(
            [
                f"{idx}. Item: {title}",
                f"   Category: {category}",
                f"   Store: {store}",
                f"   Rating: {int(interaction['rating']) if interaction['rating'].is_integer() else interaction['rating']}",
                f"   Review: {review_text}",
                f"   Time: {interaction.get('timestamp')}",
            ]
        )
    return safe_truncate("\n".join(lines), max_chars)


def build_candidate_text(item_meta: dict[str, Any], max_chars: int = 1500) -> str:
    categories = extract_categories(item_meta)
    features = extract_features(item_meta)
    details = selected_details(item_meta)
    lines = [
        "Candidate item:",
        f"Title: {item_meta.get('title') or 'unknown'}",
        f"Category: {(categories or ['unknown'])[0]}",
        f"Store: {item_meta.get('store') or 'unknown'}",
        f"Price: {item_meta.get('price') or 'unknown'}",
        f"Features: {'; '.join(features) if features else 'unknown'}",
        f"Description: {extract_description(item_meta) or 'unknown'}",
        f"Details: {'; '.join(details) if details else 'unknown'}",
    ]
    return safe_truncate("\n".join(lines), max_chars)


def valid_triples(triples: list[list[str]]) -> list[list[str]]:
    cleaned = []
    seen = set()
    for triple in triples:
        if len(triple) != 3:
            continue
        src, rel, dst = [str(part or "").strip() for part in triple]
        if not src or not rel or not dst:
            continue
        key = (src, rel, dst)
        if key not in seen:
            seen.add(key)
            cleaned.append([src, rel, dst])
    return cleaned


def add_item_metadata_triples(
    triples: list[list[str]],
    item: str,
    meta: dict[str, Any],
    max_features_per_item: int,
) -> None:
    for category in extract_categories(meta):
        triples.append([item, "belongs_to", typed_node("category", category)])
    if meta.get("store"):
        triples.append([item, "sold_by", typed_node("store", meta["store"])])
    for feature in extract_features(meta)[:max_features_per_item]:
        triples.append([item, "has_feature", typed_node("feature", feature)])
    triples.append([item, "has_price_range", typed_node("price_range", price_to_range(meta.get("price")))])


def aspect_sentiments(review_text: str) -> list[tuple[str, str]]:
    lowered = str(review_text or "").lower()
    positives = [cue for cue in POSITIVE_CUES if cue in lowered]
    negatives = [cue for cue in NEGATIVE_CUES if cue in lowered]
    results = []
    for aspect in ASPECT_VOCAB:
        needle = aspect.replace("_", " ")
        if aspect in lowered or needle in lowered:
            if positives:
                results.append(("positive_about", aspect))
            if negatives:
                results.append(("negative_about", aspect))
    return results


def build_user_behavioral_kg(
    history: list[dict[str, Any]],
    metadata_map: dict[str, dict[str, Any]],
    max_features_per_item: int = 20,
    max_aspects_per_user: int = 30,
    include_aspects: bool = True,
) -> list[list[str]]:
    user = "user:current_user"
    triples: list[list[str]] = []
    aspect_count = 0
    for interaction in history:
        item = item_node(interaction["item_id"])
        meta = metadata_map.get(interaction["item_id"], {})
        rating = int(round(float(interaction["rating"])))
        triples.append([user, "reviewed", item])
        triples.append([user, f"rated_{rating}", item])
        triples.append([item, "interacted_at", typed_node("time_bucket", timestamp_to_bucket(interaction.get("timestamp")))])
        add_item_metadata_triples(triples, item, meta, max_features_per_item)
        if include_aspects and aspect_count < max_aspects_per_user:
            for relation, aspect in aspect_sentiments(interaction.get("text", "")):
                triples.append([user, relation, typed_node("aspect", aspect)])
                aspect_count += 1
                if aspect_count >= max_aspects_per_user:
                    break
    triples = valid_triples(triples)
    if not triples:
        fallback_item = item_node(history[0]["item_id"] if history else "unknown_item")
        triples = [[user, "has_history", fallback_item]]
    return triples


def build_candidate_item_kg(
    item_meta: dict[str, Any],
    max_features_per_item: int = 20,
    item_id: str = "target_item",
) -> list[list[str]]:
    item = item_node(item_id or "target_item")
    triples: list[list[str]] = []
    add_item_metadata_triples(triples, item, item_meta, max_features_per_item)
    triples = valid_triples(triples)
    if not triples:
        triples = [[item, "has_metadata", "metadata:available"]]
    return triples


def label_for_rating(
    rating: float,
    positive_threshold: int,
    negative_threshold: int,
    drop_neutral: bool,
) -> Optional[int]:
    if rating >= positive_threshold:
        return 1
    if rating <= negative_threshold:
        return 0
    if drop_neutral:
        return None
    return 0


def split_indices(
    samples: list[dict[str, Any]],
    strategy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not samples:
        return [], [], []

    def ratio_split(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    if strategy == "temporal_global":
        ordered = sorted(range(len(samples)), key=lambda i: (samples[i]["target_timestamp"], i))
        return ratio_split(ordered)

    by_user: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_user[sample["user_id"]].append(idx)
    train, val, test = [], [], []
    for user_id in sorted(by_user):
        ordered = sorted(by_user[user_id], key=lambda i: (samples[i]["target_timestamp"], i))
        user_train, user_val, user_test = ratio_split(ordered)
        train.extend(user_train)
        val.extend(user_val)
        test.extend(user_test)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_indices(path: str, indices: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(str(i) for i in indices))


def build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    review_rows = [clean_review(row) for row in load_jsonl(args.reviews_path)]
    if args.verified_only:
        review_rows = [row for row in review_rows if bool(row.get("verified_purchase"))]
    metadata_map = build_metadata_map(load_jsonl(args.metadata_path))

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        by_user[str(row["user_id"])].append(row)

    samples = []
    for user_id in sorted(by_user):
        interactions = sorted(by_user[user_id], key=lambda row: (row["timestamp"], row["item_id"]))
        for target_pos, target in enumerate(interactions):
            history_all = interactions[:target_pos]
            if len(history_all) < args.min_user_interactions:
                continue
            label = label_for_rating(
                target["rating"],
                args.positive_threshold,
                args.negative_threshold,
                args.drop_neutral,
            )
            if label is None:
                continue
            history = history_all[-args.history_k :]
            item_meta = metadata_map.get(target["item_id"], {})
            samples.append(
                {
                    "doc_text": build_history_text(history, metadata_map, args.max_history_text_chars),
                    "claim_text": build_candidate_text(item_meta, args.max_item_text_chars),
                    "doc_kg": build_user_behavioral_kg(
                        history,
                        metadata_map,
                        args.max_features_per_item,
                        args.max_aspects_per_user,
                        include_aspects=not args.disable_aspects,
                    ),
                    "claim_kg": build_candidate_item_kg(
                        item_meta,
                        args.max_features_per_item,
                    ),
                    "label": label,
                    "user_id": user_id,
                    "target_item_id": target["item_id"],
                    "target_rating": target["rating"],
                    "target_timestamp": target["timestamp"],
                    "history_item_ids": [row["item_id"] for row in history],
                    "task_type": "amazon_binary_high_rating",
                    "_target_review_text": target.get("text", ""),
                    "_target_review_title": target.get("title", ""),
                }
            )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Amazon data for GraphCheck user simulation.")
    parser.add_argument("--reviews_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="Amazon_Beauty")
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--history_k", type=int, default=10)
    parser.add_argument("--positive_threshold", type=int, default=4)
    parser.add_argument("--negative_threshold", type=int, default=2)
    parser.add_argument("--drop_neutral", type=str_to_bool, default=True)
    parser.add_argument("--verified_only", type=str_to_bool, default=False)
    parser.add_argument("--split_strategy", choices=["temporal_global", "temporal_user"], default="temporal_global")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_history_text_chars", type=int, default=4000)
    parser.add_argument("--max_item_text_chars", type=int, default=1500)
    parser.add_argument("--max_features_per_item", type=int, default=20)
    parser.add_argument("--max_aspects_per_user", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_aspects", type=str_to_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    samples = build_samples(args)
    if not samples:
        raise ValueError("No samples were built. Try lowering --min_user_interactions or checking thresholds.")

    train_idx, val_idx, test_idx = split_indices(
        samples,
        args.split_strategy,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    output_root = os.path.join(args.output_dir, args.dataset_name)
    split_root = os.path.join(output_root, "split")
    os.makedirs(split_root, exist_ok=True)

    df = pd.DataFrame(samples).drop(columns=["_target_review_text", "_target_review_title"])
    df.to_pickle(os.path.join(output_root, f"{args.dataset_name}.pkl"))
    write_indices(os.path.join(split_root, "train_indices.txt"), train_idx)
    write_indices(os.path.join(split_root, "val_indices.txt"), val_idx)
    write_indices(os.path.join(split_root, "test_indices.txt"), test_idx)

    print(f"Wrote {len(df)} samples to {output_root}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    main()
