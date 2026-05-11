import math
import random, os
import numpy as np
import torch
import pandas as pd
import re
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error

def adjust_learning_rate(param_group, LR, epoch, args):
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_accuracy(path):
    df = pd.read_json(path, lines=True)
    label_mapping = {"support": 1, "unsupport": 0}

    y_true = df["label"].map(lambda x: label_mapping.get(str(x).lower(), 0)).tolist()
    y_pred = []

    for pred in df["pred"]:
        try:
            matches = re.findall(r"unsupport|support", str(pred).strip(), re.IGNORECASE)
            if matches:
                pred_label = label_mapping[matches[0].lower()]
            else:
                pred_label = 0
        except:
            pred_label = 0
        y_pred.append(pred_label)

    y_true = [int(label) for label in y_true]
    y_pred = [int(p) for p in y_pred]
    bACC = balanced_accuracy_score(y_true, y_pred)
    return bACC


RATING_LABELS = [1, 2, 3, 4, 5]


def parse_rating(value):
    matches = re.findall(r"\b[1-5]\b", str(value))
    if matches:
        return int(matches[0])
    return None


def is_rating_labels(labels):
    labels = [str(label) for label in labels]
    return bool(labels) and all(label in {"1", "2", "3", "4", "5"} for label in labels)


def enrich_rating_output(row):
    raw_prediction = row.get("pred", row.get("prediction", ""))
    gold_rating = parse_rating(row.get("label"))
    pred_rating = parse_rating(raw_prediction)
    row = dict(row)
    row["gold_rating"] = gold_rating
    row["pred_rating"] = pred_rating
    row["raw_prediction"] = raw_prediction
    return row


def get_rating_metrics(path):
    df = pd.read_json(path, lines=True)
    y_true = [parse_rating(value) for value in df["label"]]
    if "pred_rating" in df.columns:
        y_pred = [parse_rating(value) for value in df["pred_rating"]]
    else:
        y_pred = [parse_rating(value) for value in df["pred"]]

    valid_pairs = [(true, pred) for true, pred in zip(y_true, y_pred) if true in RATING_LABELS and pred in RATING_LABELS]
    if not valid_pairs:
        return {
            "num_samples": len(df),
            "num_valid_predictions": 0,
            "invalid_prediction_count": len(df),
            "MAE": None,
            "RMSE": None,
            "accuracy": None,
            "macro_f1": None,
            "per_rating_accuracy": {rating: None for rating in RATING_LABELS},
            "confusion_matrix": [[0 for _ in RATING_LABELS] for _ in RATING_LABELS],
        }

    true_valid = [true for true, _ in valid_pairs]
    pred_valid = [pred for _, pred in valid_pairs]
    per_rating_accuracy = {}
    for rating in RATING_LABELS:
        indices = [idx for idx, value in enumerate(true_valid) if value == rating]
        if indices:
            per_rating_accuracy[rating] = sum(1 for idx in indices if pred_valid[idx] == rating) / len(indices)
        else:
            per_rating_accuracy[rating] = None

    mse = mean_squared_error(true_valid, pred_valid)
    return {
        "num_samples": len(df),
        "num_valid_predictions": len(valid_pairs),
        "invalid_prediction_count": len(df) - len(valid_pairs),
        "MAE": mean_absolute_error(true_valid, pred_valid),
        "RMSE": math.sqrt(mse),
        "accuracy": accuracy_score(true_valid, pred_valid),
        "macro_f1": f1_score(true_valid, pred_valid, labels=RATING_LABELS, average="macro", zero_division=0),
        "per_rating_accuracy": per_rating_accuracy,
        "confusion_matrix": confusion_matrix(true_valid, pred_valid, labels=RATING_LABELS).tolist(),
    }


def print_rating_report(metrics):
    print("Amazon rating evaluation report:")
    print(f"Num samples: {metrics['num_samples']}")
    print(f"Valid predictions: {metrics['num_valid_predictions']}")
    print(f"Invalid predictions: {metrics['invalid_prediction_count']}")
    print(f"MAE: {metrics['MAE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Macro-F1: {metrics['macro_f1']}")
    print(f"Per-rating accuracy: {metrics['per_rating_accuracy']}")
    print(f"Confusion matrix labels=[1,2,3,4,5]: {metrics['confusion_matrix']}")
