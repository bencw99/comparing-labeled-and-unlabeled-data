import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from distribution import create_random_distribution, Distribution
from models import MLE, FlyingSquid
from utils import get_budgets, green_combined, cross_entropy

LABELED_BUDGETS = get_budgets((20, 200, 20))
UNLABELED_BUDGETS = [1000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--n_min", type=int, default=0)
    parser.add_argument("--n_max", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--m", type=float, default=10)
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--agg", type=str, choices=["random", "mean", "median"], default="mean")
    parser.add_argument("--low_accuracy", type=float, default=.55)
    parser.add_argument("--high_accuracy", type=float, default=.75)
    parser.add_argument("--num_trials", type=int, default=1000)
    args = parser.parse_args()

    distribution = create_random_distribution(args.m, args.low_accuracy, args.high_accuracy, args.d, epsilon=0.1, seed=123)

    args.save_path.mkdir(exist_ok=True)

    labeled_accuracies = np.zeros((len(LABELED_BUDGETS), args.num_trials, args.m))
    unlabeled_accuracies = np.zeros((len(UNLABELED_BUDGETS), args.num_trials, args.m))

    for trial in tqdm(range(args.num_trials)):
        for i, n in enumerate(UNLABELED_BUDGETS):
            L_train, Y_train = distribution.sample(n)

            unlabeled_model = FlyingSquid(agg=args.agg)
            unlabeled_model.train(L_train, Y_train, None)
            unlabeled_accuracies[i, trial] = unlabeled_model.learned_accuracies()

        for i, n in enumerate(LABELED_BUDGETS):
            L_train, Y_train = distribution.sample(n)

            labeled_model = MLE()
            labeled_model.train(L_train, Y_train, None)
            labeled_accuracies[i, trial] = labeled_model.learned_accuracies()

    np.save(args.save_path / "unlabeled_accuracies.npy", unlabeled_accuracies)
    np.save(args.save_path / "labeled_accuracies.npy", labeled_accuracies)

    labeled_losses = np.zeros(len(LABELED_BUDGETS))
    unlabeled_losses = np.zeros(len(UNLABELED_BUDGETS))
    for i_l, n_l in enumerate(tqdm(LABELED_BUDGETS)):
        current_losses = []
        for trial in range(args.num_trials):
            estimated_distribution = Distribution(labeled_accuracies[i_l, trial], {})
            current_losses.append(distribution.expectation(lambda L, Y: cross_entropy(estimated_distribution.positive_probs(L), Y)))
        labeled_losses[i_l] = np.mean(current_losses)
    for i_u, n_u in enumerate(tqdm(UNLABELED_BUDGETS)):
        current_losses = []
        for trial in range(args.num_trials):
            estimated_distribution = Distribution(unlabeled_accuracies[i_u, trial], {})
            current_losses.append(distribution.expectation(lambda L, Y: cross_entropy(estimated_distribution.positive_probs(L), Y)))
        unlabeled_losses[i_u] = np.mean(current_losses)

    np.save(args.save_path / "labeled_losses.npy", labeled_losses)
    np.save(args.save_path / "unlabeled_losses.npy", unlabeled_losses)

    green_losses = np.zeros((len(LABELED_BUDGETS), len(UNLABELED_BUDGETS)))
    green_weights = np.zeros((len(LABELED_BUDGETS), len(UNLABELED_BUDGETS)))
    for i_l, n_l in enumerate(tqdm(LABELED_BUDGETS)):
        for i_u, n_u in enumerate(tqdm(UNLABELED_BUDGETS)):
            current_losses = []
            current_weights = []
            for trial in range(args.num_trials):
                if n_l == 0:
                    estimated_accuracies, weight = unlabeled_accuracies[i_u, trial], 0.0
                elif n_u == 0:
                    estimated_accuracies, weight = labeled_accuracies[i_l, trial], 1.0
                else:
                    estimated_accuracies, weight = green_combined(labeled_accuracies[i_l, trial], unlabeled_accuracies[i_u, trial], n_l)
                estimated_distribution = Distribution(estimated_accuracies, {})
                current_losses.append(distribution.expectation(lambda L, Y: cross_entropy(estimated_distribution.positive_probs(L), Y)))
                current_weights.append(weight)
            green_losses[i_l, i_u] = np.mean(current_losses)
            green_weights[i_l, i_u] = np.mean(current_weights)

    np.save(args.save_path / "green_losses.npy", green_losses)
    np.save(args.save_path / "green_weights.npy", green_weights)

    optimal_losses = np.zeros((len(LABELED_BUDGETS), len(UNLABELED_BUDGETS)))
    optimal_weights = np.zeros((len(LABELED_BUDGETS), len(UNLABELED_BUDGETS)))

    for i_l, n_l in enumerate(tqdm(LABELED_BUDGETS)):
        for i_u, n_u in enumerate(tqdm(UNLABELED_BUDGETS)):
            best_loss = np.inf
            best_weight = None
            for weight in tqdm(np.linspace(0.0, 1.0, 101)):
                current_losses = []
                for trial in range(args.num_trials):
                    estimated_accuracies = labeled_accuracies[i_l, trial] * (1 - weight) + unlabeled_accuracies[i_u, trial] * weight
                    estimated_distribution = Distribution(estimated_accuracies, {})
                    current_losses.append(distribution.expectation(lambda L, Y: cross_entropy(estimated_distribution.positive_probs(L), Y)))
                loss = np.mean(current_losses)

                if loss < best_loss:
                    best_loss = loss
                    best_weight = weight
            optimal_losses[i_l, i_u] = best_loss
            optimal_weights[i_l, i_u] = best_weight

    np.save(args.save_path / "optimal_losses.npy", optimal_losses)
    np.save(args.save_path / "optimal_weights.npy", optimal_weights)
