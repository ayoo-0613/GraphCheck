import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from dataset.utils.dataset import KGDataset
from src.config import parse_args
from model import model_path
from src.ckpt import _reload_best_model
from src.utils import enrich_rating_output, get_accuracy, get_rating_metrics, is_rating_labels
from src.utils import print_rating_report, seed_everything
from dataset.utils.collate import collate_fn
from model.graphcheck import GraphCheck

def main(args):

    seed = args.seed
    seed_everything(seed=seed)
    print(args)

    # Data loader
    dataset = KGDataset(args.dataset_name)
    test_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Build Model
    args.llm_model_path = model_path[args.llm_model_name]
    model = GraphCheck(args=args)

    # Evaluating
    os.makedirs(f'{args.output_dir}/{args.project}', exist_ok=True)
    path = f'{args.output_dir}/{args.project}/{args.dataset_name}.csv'
    print(f'path: {path}')

    # Load Model Weights
    model = _reload_best_model(model, args)

    model.eval()
    is_amazon_rating = args.dataset_name.startswith("Amazon")
    saw_rating_labels = False
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    row_dict = dict(row)
                    if is_amazon_rating or is_rating_labels([row_dict.get("label")]):
                        saw_rating_labels = True
                        row_dict = enrich_rating_output(row_dict)
                    f.write(json.dumps(row_dict) + "\n")
            progress_bar_test.update(1)

    # Evaluating
    if is_amazon_rating or saw_rating_labels:
        print_rating_report(get_rating_metrics(path))
    else:
        bacc = get_accuracy(path)
        print(f'Test BAcc: {bacc}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    gc.collect()
