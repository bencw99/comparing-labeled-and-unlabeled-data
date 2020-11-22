import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

from models import CCFlyingSquid, CCMLE, CCGreen
from utils import get_budgets, get_real_fns

LABELED_BUDGETS = [40, 80, 120, 200, 400]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["imdb"], default="imdb")
    parser.add_argument("--agg", type=str, choices=["mean", "median"], default="median")
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=40)
    parser.add_argument("--n_max", type=int, default=200)
    parser.add_argument("--step", type=int, default=40)
    args = parser.parse_args()

    (L_train, L_test), (Y_train, Y_test) = pickle.load(open(Path("data") / f"{args.dataset}.pkl", "rb"))
    sample_fn, _ = get_real_fns(L_train, L_test, Y_train, Y_test)
    class_balance = np.mean(Y_train)

    save_path = Path("results") / args.dataset
    save_path.mkdir(exist_ok=True)

    unlabeled_model = CCFlyingSquid(agg=args.agg)
    unlabeled_model.train(L_train, None, class_balance)

    results = {}
    results["unlabeled_accuracies"] = unlabeled_model.learned_accuracies()
    results["unlabeled_score"] = f1_score(Y_test, unlabeled_model.predict_probs(L_test)>=.5)

    results["labeled_score"] = defaultdict(list)
    results["labeled_accuracies"] = defaultdict(list)

    results["green_score"] = defaultdict(list)
    results["green_accuracies"] = defaultdict(list)

    for trial in tqdm(range(args.num_trials)):
        for budget in LABELED_BUDGETS:
            L_sampled, Y_sampled = sample_fn(budget)
            labeled_model = CCMLE()
            labeled_model.train(L_sampled, Y_sampled, class_balance)

            results["labeled_score"][budget].append(f1_score(Y_test, labeled_model.predict_probs(L_test)>=.5))
            results["labeled_accuracies"][budget].append(labeled_model.learned_accuracies())

            combined_model = CCGreen(unlabeled_model, labeled_model, budget, 2 * (L_train.shape[1] - 2))
            results[f"green_score"][budget].append(f1_score(Y_test, combined_model.predict_probs(L_test)>=.5))
            results[f"green_accuracies"][budget].append(combined_model.learned_accuracies())

        pickle.dump(results, open(save_path / "combined_results.pkl", "wb"))
