import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="CDs_and_Vinyl")
    parser.add_argument("--output_dir", default="data/amazon2023/hf_smoke")
    parser.add_argument("--target_users", type=int, default=20)
    parser.add_argument("--min_interactions", type=int, default=3)
    parser.add_argument("--max_scan_reviews", type=int, default=200000)
    parser.add_argument("--max_scan_meta", type=int, default=1000000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    review_out = out_dir / f"{args.category}_reviews_sample.jsonl"
    meta_out = out_dir / f"{args.category}_meta_sample.jsonl"

    print(f"Loading review stream: raw_review_{args.category}")
    review_stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{args.category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    by_user: dict[str, list[dict]] = defaultdict(list)

    for idx, row in enumerate(review_stream):
        if idx >= args.max_scan_reviews:
            break

        user_id = row.get("user_id")
        item_id = row.get("parent_asin") or row.get("asin")
        rating = row.get("rating")
        timestamp = row.get("timestamp")

        if not user_id or not item_id or rating is None or timestamp is None:
            continue

        by_user[user_id].append(dict(row))

        dense_users = [
            u for u, rs in by_user.items()
            if len(rs) >= args.min_interactions
        ]
        if len(dense_users) >= args.target_users:
            break

    selected_users = [
        u for u, rs in by_user.items()
        if len(rs) >= args.min_interactions
    ][: args.target_users]

    selected_reviews = []
    needed_items = set()

    for user_id in selected_users:
        rows = sorted(by_user[user_id], key=lambda x: x.get("timestamp", 0))
        selected_reviews.extend(rows)
        for row in rows:
            item_id = row.get("parent_asin") or row.get("asin")
            if item_id:
                needed_items.add(item_id)

    if not selected_reviews:
        raise RuntimeError(
            "No dense users found. Increase --max_scan_reviews or lower --min_interactions."
        )

    print(f"Selected users: {len(selected_users)}")
    print(f"Selected reviews: {len(selected_reviews)}")
    print(f"Needed metadata items: {len(needed_items)}")

    write_jsonl(review_out, selected_reviews)
    print(f"Wrote reviews: {review_out}")

    print(f"Loading metadata stream: raw_meta_{args.category}")
    meta_stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{args.category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    selected_meta = []
    found_items = set()

    for idx, row in enumerate(meta_stream):
        if idx >= args.max_scan_meta:
            break

        item_id = row.get("parent_asin") or row.get("asin")
        if item_id in needed_items:
            selected_meta.append(dict(row))
            found_items.add(item_id)

        if found_items >= needed_items:
            break

    write_jsonl(meta_out, selected_meta)
    print(f"Wrote metadata: {meta_out}")
    print(f"Found metadata: {len(found_items)} / {len(needed_items)}")

    missing = sorted(needed_items - found_items)
    if missing:
        print("Warning: some metadata was not found. The builder should still use fallback triples.")
        print("First missing item ids:", missing[:10])


if __name__ == "__main__":
    main()
