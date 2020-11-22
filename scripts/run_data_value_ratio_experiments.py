import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import argparse

from models import MLE, FlyingSquid
from distribution import create_random_distribution
from utils import run_sweep, cross_entropy, get_synthetic_fns

UNLABELED_SWEEP = [(100, 900, 100), (1000, 10000, 1000)]
LABELED_SWEEP = [(10, 99, 1), (100, 998, 2), (1000, 5000, 10)]
AGGS = ["mean", "median"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--m", type=float, default=10)
    parser.add_argument("--low_accuracy", type=float, default=.55)
    parser.add_argument("--high_accuracy", type=float, default=.75)
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num_trials", type=int, default=1000)
    args = parser.parse_args()

    distribution = create_random_distribution(args.m, args.low_accuracy, args.high_accuracy, args.d, epsilon=args.epsilon, seed=123)
    sample_fn, loss_fn = get_synthetic_fns(distribution)

    unlabeled_models = {agg: FlyingSquid(agg=agg) for agg in AGGS}
    labeled_model = MLE()

    args.save_path.mkdir(exist_ok=True)

    unlabeled_losses = defaultdict(list)
    labeled_losses = []
    for trial in tqdm(range(args.num_trials)):
        for agg in AGGS:
            unlabeled_losses[agg].append(run_sweep(sample_fn, loss_fn, unlabeled_model[agg], distribution.class_balance, UNLABELED_SWEEP))
        labeled_losses.append(run_sweep(sample_fn, loss_fn, labeled_model, distribution.class_balance, LABELED_SWEEP))

        for agg in AGGS:
            np.save(args.save_path / f"unlabeled_losses_{agg}.npy", unlabeled_losses[agg])
        np.save(args.save_path / f"labeled_losses.npy", labeled_losses)
