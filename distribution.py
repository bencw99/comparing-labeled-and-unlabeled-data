"""Probability distributions over L, Y to be used by models and to generate synthetic data."""
import numpy as np
from itertools import product
from abc import ABC, abstractmethod
from utils import cross_entropy, NEGATIVE, POSITIVE

class DistributionBase(ABC):
    """A base distribution class."""

    def __init__(self, class_balance=None):
        """Set the class balance."""
        self.class_balance = class_balance or .5

    def positive_probs(self, L):
        """Return P(Y=1|L)."""
        N, _ = L.shape
        log_probs = self.log_probs(L, np.ones(N))
        log_marginal_probs = self.log_marginal_probs(L)
        return np.clip(np.exp(log_probs - log_marginal_probs), 1e-6, 1-1e-6)

    def log_marginal_probs(self, L):
        """Return P(L)."""
        N, _ = L.shape
        positive_log_probs = self.log_probs(L, np.full(N, POSITIVE))
        negative_log_probs = self.log_probs(L, np.full(N, NEGATIVE))
        return positive_log_probs + np.log(1 + np.exp(negative_log_probs - positive_log_probs))

    @abstractmethod
    def log_probs(self, L, Y):
        """Return P(L,Y)."""
        raise NotImplementedError

class CCDistribution(DistributionBase):
    """A distribution which supports class conditional accuracies, but assumes LFs are conditionally independent."""

    def __init__(self, positive_accuracies, negative_accuracies, class_balance=None):
        """Create a distribution with the given accuracies conditioned on positive and negative Y."""
        super().__init__(class_balance)
        self.positive_accuracies = positive_accuracies
        self.negative_accuracies = negative_accuracies

    def log_probs(self, L, Y):
        """Return P(L,Y)."""
        class_log_probs = np.log(self.class_balance) * (Y==POSITIVE) + np.log(1 - self.class_balance) * (Y==NEGATIVE)
        Y = np.expand_dims(Y, 1)
        positive_accuracies = np.expand_dims(self.positive_accuracies, 0)
        negative_accuracies = np.expand_dims(self.negative_accuracies, 0)
        return class_log_probs + np.sum(
            np.log(positive_accuracies) * (Y==POSITIVE) * (L==POSITIVE) + \
            np.log(1 - positive_accuracies) * (Y==POSITIVE) * (L==NEGATIVE) + \
            np.log(negative_accuracies) * (Y==NEGATIVE) * (L==NEGATIVE) + \
            np.log(1 - negative_accuracies) * (Y==NEGATIVE) * (L==POSITIVE)
        , axis=1)

class Distribution(DistributionBase):
    """A distribution for synthetic experiments matching our theoretical model.

    Supports sampling, computing expectations, and self entropy.
    """

    def __init__(self, accuracies, correlations, class_balance=None):
        """Create a distribution with the given accuracies, correlations, class balance."""
        super().__init__(class_balance)
        self.accuracies = accuracies
        self.num_labeling_functions, = accuracies.shape
        self.correlations = correlations
        self.correlations_lookup = {}
        for (idx1, idx2), correlation in correlations.items():
            assert(idx1 < idx2)
            self.correlations_lookup[idx2] = idx1, correlation
            
    def _labeling_function_probs(self, L, Y, idx):
        """Return P(L[idx]=1|Y,L[:idx])."""
        if idx in self.correlations_lookup:
            parent_idx, correlation = self.correlations_lookup[idx]

            P_joint = (self.accuracies[idx] + (self.accuracies[parent_idx] + correlation - 1) * (2 * (L[:, parent_idx]==Y) - 1)) / 2
            P_parent = (L[:, parent_idx]==Y) * self.accuracies[parent_idx] + (L[:, parent_idx]!=Y) * (1 - self.accuracies[parent_idx])
            P_cond = P_joint / P_parent
        else:
            P_cond = self.accuracies[idx]
        return P_cond * (Y==POSITIVE) + (1 - P_cond) * (Y==NEGATIVE)

    def log_probs(self, L, Y):
        """Return P(L,Y)."""
        log_probs = np.log(self.class_balance) * (Y==POSITIVE) + np.log(1 - self.class_balance) * (Y==NEGATIVE)
        for i in range(self.num_labeling_functions):
            labeling_function_probs = self._labeling_function_probs(L, Y, i)
            labeling_function_probs = np.clip(labeling_function_probs, 1e-6, 1-1e-6)
            log_probs += (L[:, i]==POSITIVE) * np.log(labeling_function_probs) + (L[:, i]==NEGATIVE) * np.log(1 - labeling_function_probs)
        return log_probs

    def sample(self, n):
        """Return a sample of size n from this distribution."""
        Y = np.random.choice(2, size=n, p=[1 - self.class_balance, self.class_balance])
        L = np.zeros((n, self.num_labeling_functions))
        for i in range(self.num_labeling_functions):
            L[:, i] = (np.random.uniform(size=n) < self._labeling_function_probs(L, Y, i))
        return L, Y

    def self_entropy(self):
        """Return H(Y|L)."""
        return self.expectation(lambda L, Y: cross_entropy(self.positive_probs(L), Y))

    def expectation(self, function):
        """Return E[function]."""
        m = len(self.accuracies)
        LY = np.array(list(product(*([[NEGATIVE, POSITIVE]] * (m + 1)))))
        L = LY[:, :m]
        Y = LY[:, -1]
        probabilities = np.exp(self.log_probs(L, Y))
        outputs = function(L, Y)
        return np.sum(probabilities * outputs)

    def correlation(self, idx1, idx2):
        """Return P(L[idx1]=L[idx2])."""
        assert(idx1 < idx2)
        if (idx1, idx2) in self.correlations:
            return self.correlations[(idx1, idx2)]
        else:
            return self.accuracies[idx1] * self.accuracies[idx2] + (1 - self.accuracies[idx1]) * (1 - self.accuracies[idx2])

def _add_correlations(accuracies, d, epsilon, random):
    assert(d <= len(accuracies) // 2)
    correlations = {}
    for i in range(d):
        min_correlation = accuracies[2*i] + accuracies[2*i+1] - 1
        max_correlation = 1 - abs(accuracies[2*i] - accuracies[2*i+1])
        if epsilon is None:
            correlations[(2*i, 2*i+1)] = random.uniform(low=min_correlation, high=max_correlation)
        else:
            base_correlation = accuracies[2*i] * accuracies[2*i+1] + (1 - accuracies[2*i]) * (1 - accuracies[2*i+1])
            correlation = base_correlation + epsilon
            correlations[(2*i, 2*i+1)] = np.clip(correlation, min_correlation, max_correlation)
    return correlations

def create_random_distribution(m, low_a, high_a, d, epsilon=None, seed=None):
    """Creates a random distribution with the given parameters."""
    if seed is None:
        random = np.random
    else:
        random = np.random.RandomState(seed)
    accuracies = random.uniform(low=low_a, high=high_a, size=m)
    correlations = _add_correlations(accuracies, d, epsilon, random)
    return Distribution(accuracies, correlations)
