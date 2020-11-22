import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

from models import CCFlyingSquid, CCMLE
from utils import run_sweep, cross_entropy, get_real_fns

UNLABELED_SWEEP = [(100, 900, 100), (1000, 9000, 1000), (10000, 40000, 10000)]
LABELED_SWEEP = [(10, 98, 2), (100, 900, 100), (1000, 9000, 1000), (10000, 40000, 10000)]
AGGS = ["mean", "median"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["imdb"], default="imdb")
    parser.add_argument("--num_trials", type=int, default=1000)
    args = parser.parse_args()

    (L_train, L_test), (Y_train, Y_test) = pickle.load(open(Path("data") / f"{args.dataset}.pkl", "rb"))
    sample_fn, loss_fn = get_real_fns(L_train, L_test, Y_train, Y_test)
    class_balance = np.mean(Y_train)

    save_path = Path("results") / args.dataset
    save_path.mkdir(exist_ok=True)

    unlabeled_models = {agg: CCFlyingSquid(agg=agg, fast=True) for agg in AGGS}
    labeled_model = CCMLE()

    unlabeled_losses = defaultdict(list)
    labeled_losses = []
    for trial in tqdm(range(args.num_trials)):
        for agg in AGGS:
            unlabeled_losses[agg].append(run_sweep(sample_fn, loss_fn, unlabeled_models[agg], class_balance, UNLABELED_SWEEP))
        labeled_losses.append(run_sweep(sample_fn, loss_fn, labeled_model, class_balance, LABELED_SWEEP))

        for agg in AGGS:
            np.save(save_path / f"unlabeled_losses_{agg}.npy", unlabeled_losses[agg])
        np.save(save_path / f"labeled_losses.npy", labeled_losses)
