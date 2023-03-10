from typing import Dict
import numpy as np


class Regularizer:

    def __init__(self):
        pass

    def value(self, weights: np.ndarray) -> float:
        raise NotImplemented("Override this method!")

    def grad(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplemented("Override this method!")