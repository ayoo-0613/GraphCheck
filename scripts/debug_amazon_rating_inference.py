#!/usr/bin/env python3
"""Debug one constrained Amazon rating inference batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from argparse import Namespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import model_path
from model.graphcheck import GraphCheck


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Amazon rating inference debug batch.")
    parser.add_argument("--dataset_name", default="Amazon_CDs_HF_Rating_Smoke")
    parser.add_argument("--kg_root", default="dataset/extracted_KG")
    parser.add_argument("--llm_model_name", default="qwen_0_5b")
    parser.add_argument("--gnn_model_name", default="gt")
    parser.add_argument("--max_txt_len", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    os.environ["GRAPHCHECK_KG_ROOT"] = str((REPO_ROOT / args.kg_root).resolve())
    from dataset.utils.collate import collate_fn
    from dataset.utils.dataset import KGDataset

    train_args = Namespace(
        train_dataset=args.dataset_name,
        llm_model_name=args.llm_model_name,
        llm_model_path=model_path[args.llm_model_name],
        llm_num_virtual_tokens=4,
        max_txt_len=args.max_txt_len,
        max_new_tokens=args.max_new_tokens,
        gnn_model_name=args.gnn_model_name,
        gnn_num_layers=3,
        gnn_in_dim=1024,
        gnn_hidden_dim=1024,
        gnn_num_heads=4,
        gnn_dropout=0.3,
        max_memory=[80, 80],
    )

    dataset = KGDataset(args.dataset_name)
    batch = collate_fn([dataset[0]])
    model = GraphCheck(args=train_args)
    model.eval()
    with torch.no_grad():
        output = model.inference(batch)

    valid = {"1", "2", "3", "4", "5"}
    assert all(str(pred) in valid for pred in output["pred"]), output["pred"]
    print(f"Amazon rating inference debug passed: {output['pred']}")


if __name__ == "__main__":
    main()
