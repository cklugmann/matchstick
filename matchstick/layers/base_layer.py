from typing import Dict, Optional
import numpy as np

from matchstick.losses.regularizer import Regularizer


class BaseLayer:

    params: Dict[str, np.ndarray]
    cache: Dict[str, np.ndarray]

    def __init__(self, regularizer: Optional[Dict[str, Regularizer]] = None):
        self.grads = dict()
        if regularizer is None:
            self.regularizer = dict()
        else:
            self.regularizer = regularizer

    def build(self, *args, **kwargs) -> None:
        raise NotImplemented("Override this method!")

    @property
    def _dz(self):
        raise NotImplemented("Override this method!")

    @property
    def _dtheta(self):
        raise NotImplemented("Override this method!")

    @property
    def regularization_loss(self) -> float:
        reg_losses = list()
        for param_name, reg_obj in self.regularizer.items():
            reg_val = reg_obj.value(self.params[param_name])
            reg_losses.append(reg_val)
        return np.sum(reg_losses).item()

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplemented("Override this method!")

    def backward(self, upstream_grad: np.ndarray):
        """
        :param upstream_grad: Upstream gradient of shape (M, out_features)
        :return: Returns downstream gradients.
        """
        for param_name, param in self.params.items():
            self.grads[param_name] = np.einsum("mk, mk...", upstream_grad, self._dtheta[param_name])
            reg_obj = self.regularizer.get(param_name)
            if reg_obj is not None:
                self.grads[param_name] += reg_obj.grad(param)

        downstream_grad = np.einsum("mk, mk... -> m...", upstream_grad, self._dz)
        self.cache.clear()

        return downstream_grad
