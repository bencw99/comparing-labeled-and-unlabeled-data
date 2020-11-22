import numpy as np
import sympy
from scipy.optimize import fsolve
from abc import ABC, abstractmethod

from distribution import Distribution, CCDistribution
from utils import NEGATIVE, POSITIVE, green_combined

def _get_agg(agg_name):
    if agg_name == "mean":
        return np.mean
    elif agg_name == "median":
        return np.median
    elif agg_name == "random":
        return np.random.choice
    else:
        raise ValueError

class ModelBase(ABC):
    def __init__(self):
        self.distribution = None

    @abstractmethod
    def train(self, L, labels, class_balance):
        raise NotImplementedError

    def predict_probs(self, L):
        if self.distribution is None:
            return None
        return self.distribution.positive_probs(L)

    @abstractmethod
    def learned_accuracies(self):
        raise NotImplementedError

class CCModelBase(ModelBase, ABC):
    def __init__(self):
        self.distributon = None

    def learned_accuracies(self):
        if self.distribution is None:
            return None
        return np.stack([
            self.distribution.negative_accuracies,
            self.distribution.positive_accuracies,
        ], axis=-1)

class CCFlyingSquid(CCModelBase):
    def __init__(self, agg="median", fast=False, fast_max_attempts=10):
        super().__init__()
        self.agg = _get_agg(agg)
        self.fast = fast
        self.fast_max_attempts = fast_max_attempts

    def _compute_accuracies(self, L, class_balance):
        N, M = L.shape
        d = class_balance / (1 - class_balance)
        c = np.mean(L==POSITIVE, axis=0) / (1 - class_balance)
        O = L.T @ L / N

        def _equation(i, j, alpha, beta):
            return d * (1 + d) * alpha * beta + c[i] * c[j] - c[i] * d * beta - c[j] * d * alpha - O[i, j] / (1 - class_balance)

        def _triplet_equations(p):
            alpha, beta, gamma = p
            return [
                _equation(i, j, alpha, beta),
                _equation(j, k, beta, gamma),
                _equation(i, k, alpha, gamma),
            ]

        def _fast_solve_triplet(i, j, k):
            first_root = fsolve(_triplet_equations, [1.0, 1.0, 1.0])
            if np.allclose(_triplet_equations(first_root), [0.0, 0.0, 0.0]):
                second_root = fsolve(_triplet_equations, np.random.uniform(size=3))
                attempts = 0
                while np.allclose(second_root, first_root):
                    second_root = fsolve(_triplet_equations, np.random.uniform(size=3))
                    attempts += 1
                    if attempts > self.fast_max_attempts:
                        break
                return (max(float(first_root[i]), float(second_root[i])) for i in range(3))

        def _full_solve_triplet(i, j, k):
            p = sympy.symbols("alpha beta gamma")
            solutions = sympy.solve(_triplet_equations(p), *p)
            if isinstance(solutions[0][0], sympy.core.numbers.Float):
                return (max(float(solutions[0][i]), float(solutions[1][i])) for i in range(3))
        
        _solve_triplet = _fast_solve_triplet if self.fast else _full_solve_triplet
        alphas = [[] for i in range(M)]
        for i in range(M):
            for j in range(i + 1, M):
                for k in range(j + 1, M):
                    solution = _solve_triplet(i, j, k)
                    if solution is not None:
                        alpha_i, alpha_j, alpha_k = solution
                        alphas[i].append(alpha_i)
                        alphas[j].append(alpha_j)
                        alphas[k].append(alpha_k)

        positive_accuracies = np.clip([self.agg(alphas[i]) for i in range(M)], 1/N, 1-1/N)
        negative_accuracies = np.clip(1 - (c - d * positive_accuracies), 1/N, 1-1/N)
        return positive_accuracies, negative_accuracies

    def train(self, L, labels, class_balance):
        positive_accuracies, negative_accuracies = self._compute_accuracies(L, class_balance)
        self.distribution = CCDistribution(positive_accuracies, negative_accuracies, class_balance)

class CCMLE(CCModelBase):
    def train(self, L, labels, class_balance):
        N, _ = L.shape
        positive_accuracies = np.clip(np.mean(L[labels==POSITIVE]==POSITIVE, axis=0), 1/N, 1-1/N)
        negative_accuracies = np.clip(np.mean(L[labels==NEGATIVE]==NEGATIVE, axis=0), 1/N, 1-1/N)
        self.distribution = CCDistribution(
            positive_accuracies,
            negative_accuracies,
            class_balance
        )

class CCCombined(CCModelBase):
    def __init__(self, flying_squid, mle, weight):
        positive_accuracies = flying_squid.distribution.positive_accuracies * (1 - weight) + \
            mle.distribution.positive_accuracies * weight
        negative_accuracies = flying_squid.distribution.negative_accuracies * (1 - weight) + \
            mle.distribution.negative_accuracies * weight
        class_balance = flying_squid.distribution.class_balance * (1 - weight) + \
            mle.distribution.class_balance * weight
        self.distribution = CCDistribution(
            positive_accuracies,
            negative_accuracies,
            class_balance
        )

    def train(self, L, labels, class_balance):
        raise NotImplementedError

class CCGreen(CCCombined):
    def __init__(self, flying_squid, mle, n_l, R=None):
        unlabeled_accuracies = flying_squid.learned_accuracies()
        labeled_accuracies = mle.learned_accuracies()
        _, weight = green_combined(labeled_accuracies, unlabeled_accuracies, n_l, R=R)
        super().__init__(flying_squid, mle, 1 - weight)

class SimpleModelBase(ModelBase, ABC):
    def __init__(self):
        self.distributon = None

    def learned_accuracies(self):
        if self.distribution is None:
            return None
        return self.distribution.accuracies

class FlyingSquid(SimpleModelBase):
    def __init__(self, agg="median"):
        super().__init__()
        self.agg = _get_agg(agg)

    def train(self, L, labels, class_balance):
        N, M = L.shape

        if N == 0:
            accuracies = np.full(M, .5)
        else:
            c = (2 * L.T - 1) @ (2 * L - 1) / N
            c[c==0] = 1 / N
            a = [[] for i in range(M)]
            for i in range(M):
                for j in range(M - 1):
                    for k in range(j + 1, M - 1):
                        j_temp = j + 1 if j >= i else j
                        k_temp = k + 1 if k >= i else k
                        temp = c[i, j_temp] * c[i, k_temp] / c[j_temp, k_temp]
                        if temp > 0:
                            a[i].append(np.clip(np.sqrt(temp), -1+1/N, 1-1/N))
            a = np.array([self.agg(a[i]) for i in range(M)])
            accuracies = (a + 1) / 2
        self.distribution = Distribution(accuracies, {}, class_balance=class_balance)

class MLE(SimpleModelBase):
    def train(self, L, labels, class_balance):
        N, M = L.shape

        if N == 0:
            accuracies = np.full(M, .5)
        else:
            corrects = L==np.expand_dims(labels, 1)
            accuracies = np.mean(corrects, axis=0)
            covariance = (corrects - accuracies)
            accuracies = np.clip(accuracies, 1/N, 1-1/N)
        self.distribution = Distribution(accuracies, {}, class_balance=class_balance)
