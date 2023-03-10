from typing import List, Dict, Optional, Callable
import numpy as np

from matchstick.layers import BaseLayer
from matchstick.losses import Loss


class Sequential:
    def __init__(self, layers: List[BaseLayer]):
        self.layers = layers
        self.cache = dict()
        self.loss = None
        self.loss_value = None

    @property
    def params(self) -> Dict[str, np.ndarray]:
        out = dict()
        for idx, layer in enumerate(self.layers):
            for param_name, param in layer.params.items():
                out[f"{idx}:{param_name}"] = param
        return out

    @property
    def regularization_loss(self):
        reg_losses = list()
        for layer in self.layers:
            reg_losses.append(layer.regularization_loss)
        return np.sum(reg_losses).item()

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        out = dict()
        for idx, layer in enumerate(self.layers):
            for param_name, param in layer.grads.items():
                out[f"{idx}:{param_name}"] = param
        return out

    def compile(self, loss: Loss, in_features: int) -> None:
        self.loss = loss
        for layer in self.layers:
            layer.build(in_features=in_features)
            if hasattr(layer, "out_features"):
                in_features = layer.out_features

    def clear(self) -> None:
        self.cache.clear()
        self.loss.clear()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input_batch: np.ndarray) -> np.ndarray:
        out = input_batch
        for layer in self.layers:
            out = layer.forward(out)
        self.cache["pred"] = out.copy()
        return out

    def compute_loss(
        self,
        target: np.ndarray,
        backprop: bool = True,
        callbacks: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, float]:

        if "pred" not in self.cache:
            raise ValueError("You have to perform the forward pass first.")

        if callbacks is None:
            callbacks = dict()

        def dummy(*args, **kwargs):
            pass

        pred = self.cache["pred"]
        self.loss_value = self.loss.forward(pred, target)

        if backprop:
            upstream_gradient = self.loss.backward()
            layer_idx = len(self.layers)
            callbacks.get("before_backprop", dummy)(layer_idx, upstream_gradient)
            for layer in self.layers[::-1]:
                callbacks.get("before_layer_backprop", dummy)(layer_idx, upstream_gradient)
                upstream_gradient = layer.backward(upstream_gradient)
                callbacks.get("after_layer_backprop", dummy)(layer_idx, upstream_gradient)
                layer_idx -= 1
            callbacks.get("after_backprop", dummy)(layer_idx, upstream_gradient)
            self.clear()

        return {
            "main": self.loss_value,
            "regularization": self.regularization_loss,
            "total": self.loss_value + self.regularization_loss,
        }
