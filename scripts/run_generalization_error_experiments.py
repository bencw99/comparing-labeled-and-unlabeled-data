import numpy as np
import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path
from tqdm import tqdm
import argparse

from models import FlyingSquid, MLE
from distribution import create_random_distribution
from utils import cross_entropy

config = {
    "m": [10],
    "low_a": [.55],
    "high_a": [.75],
    "d": list(range(10)),
    "epsilon": [0.1],
    "n": [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
    "mode": ["labeled", "mean", "median"],
}

def run_experiment(m, low_a, high_a, d, epsilon, n, mode):
    distribution = create_random_distribution(m, low_a, high_a, d, epsilon, seed=123)

    L_train, Y_train = distribution.sample(n)
    model = MLE() if mode == "labeled" else FlyingSquid(mode)
    model.train(L_train, Y_train, distribution.class_balance)

    loss = lambda L, Y: cross_entropy(model.predict_probs(L), Y)
    return distribution.expectation(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_trials", type=int, default=1000)
    args = parser.parse_args()

    results = defaultdict(list)
    save_path = Path(args.save_path)
    if save_path.exists():
        existing_results = pickle.load(open(save_path, "rb"))
    else:
        existing_results = {}
    results.update(existing_results)
    for trial in tqdm(range(args.num_trials)):
        keys, values = zip(*config.items())
        for bundle in product(*values):
            params = dict(zip(keys, bundle))
            if params["d"] > params["m"] // 2:
                continue
            if str(params) in existing_results:
                continue
            loss = run_experiment(**params)
            results[str(params)].append(loss)
        pickle.dump(results, open(save_path, "wb"))
