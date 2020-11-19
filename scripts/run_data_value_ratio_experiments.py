import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import argparse

from models import MLE, FlyingSquid
from distribution import create_random_distribution
from utils import cross_entropy, get_budgets

def _run_sweep(distribution, model, sweep):
    ns = get_budgets(sweep)
    losses = np.zeros(ns.shape)
    for i, n in enumerate(ns):
        L, Y = distribution.sample(n)
        model.train(L, Y, distribution.class_balance)
        losses[i] = distribution.expectation(lambda L, Y: cross_entropy(model.predict_probs(L), Y))
    return losses

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

    unlabeled_models = {agg: FlyingSquid(agg=agg) for agg in AGGS}
    labeled_model = MLE()

    args.save_path.mkdir(exist_ok=True)

    unlabeled_losses = defaultdict(list)
    labeled_losses = []
    for trial in tqdm(range(args.num_trials)):
        for agg in AGGS:
            unlabeled_losses[agg].append(_run_sweep(distribution, unlabeled_models[agg], UNLABELED_SWEEP))
        labeled_losses.append(_run_sweep(distribution, labeled_model, LABELED_SWEEP))

        for agg in AGGS:
            np.save(args.save_path / f"unlabeled_losses_{agg}.npy", unlabeled_losses[agg])
        np.save(args.save_path / f"labeled_losses.npy", labeled_losses)
