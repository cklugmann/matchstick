from typing import Optional, Dict
import numpy as np

from matchstick.layers.base_layer import BaseLayer


class Linear(BaseLayer):
    def __init__(
        self,
        out_features: int,
        in_features: Optional[int] = None,
        bias: bool = True,
        activation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.bias = bias
        self.activation = activation
        self.params = dict()
        self.cache = dict()
        self.build(in_features=in_features)

    @property
    def in_features(self) -> int:
        if not self.is_built():
            raise ValueError("Model not built yet.")
        *_, in_features = self.params["weights"].shape
        return in_features

    def is_built(self) -> bool:
        return "weights" in self.params

    def build(self, in_features: Optional[int] = None) -> None:
        if self.bias and "bias" not in self.params:
            self.params["bias"] = np.zeros((self.out_features,), dtype=np.float32)
        if in_features is not None:
            self.params["weights"] = np.random.uniform(low=-1, high=1, size=(self.out_features, in_features))

    def activation_function(self, inputs: np.ndarray) -> np.ndarray:
        out = inputs
        if self.activation:
            out = np.maximum(out, 0)
        return out

    def forward(self, input_batch: np.ndarray) -> np.ndarray:

        _, in_features = input_batch.shape
        if not self.is_built():
            self.build(in_features=in_features)

        out = np.einsum(
            "ij, ...j -> ...i", self.params["weights"], input_batch
        )
        if self.bias:
            out = out + self.params["bias"]

        self.cache["pre_activation"] = out.copy()

        out = self.activation_function(out)

        self.cache["previous_activation"] = input_batch.copy()
        return out

    @property
    def _dz(self) -> np.ndarray:
        if "previous_activation" not in self.cache:
            raise ValueError("Nothing in cache.")

        # Input feature tensor of shape (M, in_features)
        Z = self.cache["previous_activation"]

        # Goal: produce (M, out_features, in_features)

        # Pre activation of shape (M, out_features)
        dz = self.cache["pre_activation"]
        if self.activation:
            dz = (dz > 0).astype(Z.dtype)
        else:
            dz = np.ones_like(dz)

        dz = np.expand_dims(dz, -1) * np.expand_dims(self.params["weights"], 0)

        return dz

    @property
    def _dtheta(self) -> Dict[str, np.ndarray]:
        if "previous_activation" not in self.cache:
            raise ValueError("Nothing in cache.")

        # Input feature tensor of shape (M, in_features)
        Z = self.cache["previous_activation"]
        batch_size, *_ = Z.shape

        # Goal: produce (M, out_features, *param_dims)

        # Pre activation of shape (M, out_features)
        d = self.cache["pre_activation"]
        if self.activation:
            d = (d > 0).astype(Z.dtype)
        else:
            d = np.ones_like(d)

        unit_matrix = np.eye(self.out_features, dtype=d.dtype)

        dtheta = dict()
        dtheta["weights"] = (
            d.reshape(batch_size, self.out_features, 1, 1)
            * unit_matrix.reshape(1, self.out_features, self.out_features, 1)
            * Z.reshape(batch_size, 1, 1, self.in_features)
        )
        if self.bias:
            dtheta["bias"] = np.expand_dims(d, -1) * np.expand_dims(unit_matrix, 0)

        return dtheta
