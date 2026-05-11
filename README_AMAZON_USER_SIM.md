# Amazon User Simulation Adaptation

## Purpose

This extension adapts GraphCheck from KG-powered fact checking to Amazon behavioral-KG-conditioned user simulation. Amazon user simulation is now formulated as exact 1-to-5 rating prediction.

The original GraphCheck fact-checking datasets still use `support` / `unsupport`. Amazon datasets do not use `support` / `unsupport`; their labels are `"1"`, `"2"`, `"3"`, `"4"`, or `"5"`.

## Mapping

| GraphCheck | Amazon user simulation |
| --- | --- |
| `doc_text` | user history text before the target interaction |
| `claim_text` | candidate item metadata text |
| `doc_kg` | user behavioral KG from historical interactions |
| `claim_kg` | candidate item KG from metadata |
| `label` | exact target rating, one of 1, 2, 3, 4, 5 |

Input: user history text, candidate item metadata text, user behavioral KG, and candidate item KG.

Output: exact user rating from 1 to 5.

The target review title/text is future information and is never used in `doc_text`, `claim_text`, `doc_kg`, or `claim_kg`.

## Data Format

The builder reads Amazon Reviews 2023 style JSONL.

Review fields:
`user_id`, `parent_asin` or `asin`, `rating`, `title`, `text`, `timestamp`, `verified_purchase`.

Metadata fields:
`parent_asin` or `asin`, `title`, `main_category`, `categories`, `features`, `description`, `price`, `store`, `details`, `bought_together`.

## Build Amazon Dataset

```bash
python scripts/build_amazon_graphcheck_data.py \
  --reviews_path /path/to/reviews.jsonl \
  --metadata_path /path/to/meta.jsonl \
  --output_dir dataset/extracted_KG \
  --dataset_name Amazon_Beauty \
  --min_user_interactions 5 \
  --history_k 10
```

Outputs:

- `dataset/extracted_KG/Amazon_Beauty/Amazon_Beauty.pkl`
- `dataset/extracted_KG/Amazon_Beauty/split/train_indices.txt`
- `dataset/extracted_KG/Amazon_Beauty/split/val_indices.txt`
- `dataset/extracted_KG/Amazon_Beauty/split/test_indices.txt`

Deprecated binary arguments such as `--positive_threshold`, `--negative_threshold`, and `--drop_neutral` are accepted for CLI compatibility but ignored. Rating 3 is retained.

## Build Graphs

```bash
python graph_build.py --data_name Amazon_Beauty
```

If split files already exist, `graph_build.py` reuses them and does not overwrite temporal splits.

## Prompt

Amazon datasets use this rating prompt:

```text
Question: Based on the user's historical behavior, predict the rating that the user would give to the candidate item.
Please answer with one number only: 1, 2, 3, 4, or 5.

User History:
{doc_text}

Candidate Item:
{claim_text}

Rating:
```

Non-Amazon datasets keep the original GraphCheck fact-checking prompt.

## Train

```bash
python train.py \
  --project Amazon_Beauty_GraphUserSim \
  --train_dataset Amazon_Beauty \
  --llm_model_name qwen_7b \
  --gnn_model_name gt \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_epochs 10
```

Full training is best run on a GPU server. Local smoke tests stop before full training by default.

## Evaluate

GraphCheck training and inference write JSONL rows with `label` and `pred` or `prediction`. Compute rating metrics with:

```bash
python scripts/evaluate_amazon_rating.py \
  --input_path output/Amazon_Beauty_GraphUserSim/validation.csv \
  --pred_col pred
```

Metrics include MAE, RMSE, accuracy, macro F1, per-rating accuracy, invalid prediction count, and confusion matrix.

`scripts/evaluate_amazon_binary.py` is deprecated and exits with an error.

## KG-Text Baseline

Create text-only prompts that serialize both KGs:

```bash
python scripts/build_amazon_kg_text_prompt.py \
  --input_pkl dataset/extracted_KG/Amazon_Beauty/Amazon_Beauty.pkl \
  --output_path output/amazon_beauty_kg_text_prompts.jsonl
```

This supports later comparison among raw-history LLM prompting, KG-text prompting, and GraphCheck GNN/projector prompting.

## Automated Smoke Test Pipeline

Run the smallest fixture-based local test without HuggingFace access:

```bash
python scripts/smoke_test_amazon_pipeline.py
```

Run an official HuggingFace Amazon Reviews 2023 small sample pipeline:

```bash
python scripts/run_amazon_smoke_pipeline.py \
  --category CDs_and_Vinyl \
  --dataset_name Amazon_CDs_HF_Rating_Smoke \
  --target_users 20 \
  --min_interactions 3 \
  --max_scan_reviews 200000 \
  --min_user_interactions 2 \
  --history_k 3
```

Expected outputs:

- `dataset/extracted_KG/Amazon_CDs_HF_Rating_Smoke/Amazon_CDs_HF_Rating_Smoke.pkl`
- `dataset/extracted_KG/Amazon_CDs_HF_Rating_Smoke/split/train_indices.txt`
- `dataset/extracted_KG/Amazon_CDs_HF_Rating_Smoke/split/val_indices.txt`
- `dataset/extracted_KG/Amazon_CDs_HF_Rating_Smoke/split/test_indices.txt`
- `dataset/extracted_KG/Amazon_CDs_HF_Rating_Smoke/graphs/doc/*.pt`
- `dataset/extracted_KG/Amazon_CDs_HF_Rating_Smoke/graphs/claim/*.pt`

The pipeline samples reviews, retrieves matching metadata, builds the rating pkl, runs `graph_build.py`, validates graph files, validates `KGDataset`, and validates `collate_fn`.

By default, local macOS smoke runs stop before `train.py`. For a tiny train check, use `--run_train_smoke`, which defaults to `qwen_0_5b`, batch size 1, and short generation.

## Leakage Prevention Rules

- Target review text and title are not included in any model input.
- History is strictly interactions before the target timestamp.
- Candidate input uses metadata only: title, category, features, description, store, price, and details.
- Temporal splits are generated by the Amazon builder and preserved by `graph_build.py`.

## Current Limitations

- The current Amazon task is exact 1-to-5 rating prediction only.
- The earlier binary high-rating Amazon task is deprecated and not used in the current pipeline.
- Aspect extraction is deterministic keyword matching, not a learned extractor.
- User nodes are sample-local and anonymized as `user:current_user`.
- Candidate and history KGs are homogeneous GraphCheck triple lists, not heterogeneous PyG `HeteroData`.

## Future Extensions

- Review aspect prediction.
- Personalized explanation generation.
- Heterogeneous user/item graphs with HGT.
- Multi-token graph soft prompt projectors.
