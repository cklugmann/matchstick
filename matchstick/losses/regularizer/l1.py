import numpy as np

from matchstick.losses.regularizer.regularizer import Regularizer


class L1(Regularizer):

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.cache = dict()

    def value(self, weights: np.ndarray) -> float:
        self.cache["weights"] = weights
        return self.scale * np.sum(np.abs(weights))

    def grad(self, weights: np.ndarray) -> np.ndarray:
        is_positive = (weights > 0).astype(weights.dtype)
        res = self.scale * (2 * is_positive - 1)
        return res
