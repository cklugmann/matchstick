from typing import Dict
import numpy as np


class Adam:
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        learning_rate: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-5
    ):
        self.params = params
        self.learning_rate = learning_rate
        self.cache = dict()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0

    def apply(self, grads: Dict[str, np.ndarray]) -> None:
        for param_name, param in self.params.items():

            grad = grads[param_name]
            previous_m = self.cache.get(f"{param_name}/m", np.zeros_like(grad))
            previous_v = self.cache.get(f"{param_name}/v", np.zeros_like(grad))
            self.step += 1

            new_m = self.beta1 * previous_m + (1 - self.beta1) * grad
            new_v = self.beta2 * previous_v + (1 - self.beta2) * (grad ** 2)

            self.cache[f"{param_name}/m"] = new_m.copy()
            self.cache[f"{param_name}/v"] = new_v.copy()

            new_m /= (1 - self.beta1 ** self.step)
            new_v /= (1 - self.beta2 ** self.step)

            update = new_m / (np.sqrt(new_v) + self.eps)
            param -= self.learning_rate * update
