import numpy as np

POSITIVE = 1
NEGATIVE = 0

def cross_entropy(Y_hat, Y):
    return -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

def get_budgets(sweep):
    if isinstance(sweep[0], int):
        n_min, n_max, step = sweep
        return np.linspace(n_min, n_max, (n_max - n_min) // step + 1, dtype=np.int32)
    else:
        return np.concatenate([get_budgets(sub_sweep) for sub_sweep in sweep])

def green_combined(labeled_accuracies, unlabeled_accuracies, n_l, R=None):
    R = R or labeled_accuracies.shape[0] - 2
    temp = labeled_accuracies - unlabeled_accuracies
    denom = np.sum((temp ** 2) / (labeled_accuracies * (1 - labeled_accuracies) / n_l))
    weight = np.maximum(1 - R / denom, 0)
    return unlabeled_accuracies + weight * temp, (1 - weight)

def get_synthetic_fns(distribution):
    return distribution.sample, lambda model: distribution.expectation(lambda L, Y: cross_entropy(model.predict_probs(L), Y))

def get_real_fns(L_train, L_test, Y_train, Y_test):
    def sample_fn(n):
        N, _ = L_train.shape
        sample_idxs = np.zeros(N, dtype=np.bool)
        sample_idxs[:n] = True
        np.random.shuffle(sample_idxs)
        return L_train[sample_idxs], Y_train[sample_idxs]

    def loss_fn(model):
        return np.mean(cross_entropy(model.predict_probs(L_test), Y_test))

    return sample_fn, loss_fn

def run_experiment(sample_fn, loss_fn, model, class_balance, n):
    L, Y = sample_fn(n)
    model.train(L, Y, class_balance)
    return loss_fn(model)

def run_sweep(sample_fn, loss_fn, model, class_balance, sweep):
    ns = get_budgets(sweep)
    losses = np.zeros(ns.shape)
    for i, n in enumerate(ns):
        losses[i] = run_experiment(sample_fn, loss_fn, model, class_balance, n)
    return losses

def _ensure_descending(losses):
    """Ensures that losses decrease with larger n (might not be the case due to sampling noise)."""

    for i in range(len(losses) - 1):
        if losses[i + 1] > losses[i]:
            losses[i + 1] = losses[i]
    return losses

def data_value_ratios(unlabeled_losses, n_us, labeled_losses, n_ls):
    """Computes the data value ratio for the given unlabeled and labeled losses."""

    mean_unlabeled_losses = _ensure_descending(np.nanmean(unlabeled_losses, 0))
    mean_labeled_losses = _ensure_descending(np.nanmean(labeled_losses, 0))

    data_value_ratios = []
    i_l = 0

    for i_u, n_u in enumerate(n_us):
        alpha = 0
        while mean_labeled_losses[i_l] > mean_unlabeled_losses[i_u]:
            i_l += 1
        data_value_ratios.append(n_u / n_ls[i_l - 1])
    return data_value_ratios
