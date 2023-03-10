from typing import Dict
import numpy as np


class SGD:
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        learning_rate: float = 1e-1,
        momentum: float = 0.9,
    ):
        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cache = dict()

    def apply(self, grads: Dict[str, np.ndarray]) -> None:
        for param_name, param in self.params.items():
            grad = grads[param_name]
            previous_grad = self.cache.get(param_name, grad)
            update = (
                self.momentum * previous_grad + (1 - self.momentum) * grad
            )
            self.cache[param_name] = update.copy()
            param -= self.learning_rate * update
