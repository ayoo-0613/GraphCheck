#!/usr/bin/env python3
"""Run the fixture-based Amazon pipeline smoke test without HuggingFace access."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Amazon fixture smoke pipeline.")
    parser.add_argument("--output_dir", default="dataset/extracted_KG")
    parser.add_argument("--dataset_name", default="Amazon_Test_Rating")
    parser.add_argument("--skip_graph_build", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_amazon_smoke_pipeline.py"),
        "--category",
        "Amazon_Test",
        "--dataset_name",
        args.dataset_name,
        "--output_dir",
        args.output_dir,
        "--hf_output_dir",
        "data/amazon2023/fixture_smoke",
        "--reviews_path",
        str(REPO_ROOT / "tests" / "fixtures" / "amazon_reviews_small.jsonl"),
        "--metadata_path",
        str(REPO_ROOT / "tests" / "fixtures" / "amazon_meta_small.jsonl"),
        "--min_user_interactions",
        "5",
        "--history_k",
        "5",
        "--target_users",
        "3",
        "--min_interactions",
        "5",
    ]
    if args.skip_graph_build:
        cmd.append("--skip_graph_build")
    if args.force_rebuild:
        cmd.append("--force_rebuild")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print("Amazon rating smoke test passed")


if __name__ == "__main__":
    main()
