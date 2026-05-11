#!/usr/bin/env python3
"""Create a text-only KG prompt baseline from an Amazon GraphCheck pkl."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def serialize_triples(triples: list[list[str]]) -> str:
    return "\n".join(f"({src}, {rel}, {dst})" for src, rel, dst in triples)


def build_prompt(row: pd.Series) -> str:
    return (
        "Question: Based on the user's historical behavior and KGs, predict the rating that the user would give to the candidate item.\n"
        "Please answer with one number only: 1, 2, 3, 4, or 5.\n\n"
        f"User History:\n{row['doc_text']}\n\n"
        f"Candidate Item:\n{row['claim_text']}\n\n"
        f"User Behavioral KG:\n{serialize_triples(row['doc_kg'])}\n\n"
        f"Candidate Item KG:\n{serialize_triples(row['claim_kg'])}\n\n"
        "Rating:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Amazon KG-text prompt baseline JSONL.")
    parser.add_argument("--input_pkl", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    df = pd.read_pickle(args.input_pkl)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for idx, row in df.iterrows():
            record = {
                "id": int(idx),
                "prompt": build_prompt(row),
                "label": str(int(row["label"])),
                "user_id": row.get("user_id"),
                "target_item_id": row.get("target_item_id"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df)} prompts to {args.output_path}")


if __name__ == "__main__":
    main()
