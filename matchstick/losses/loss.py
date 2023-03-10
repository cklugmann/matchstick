from typing import Dict
import numpy as np


class Loss:

    cache: Dict[str, np.ndarray]

    def __init__(self, reduce: str = "mean"):
        allowed_reduce = ["sum", "mean", "no"]
        if reduce not in allowed_reduce:
            raise ValueError("`{}` is not proper reduction method.")
        self._reduce = reduce
        self._batch_size = None

    def clear(self) -> None:
        self.cache.clear()

    def reduce_function(self, input_batch: np.ndarray):
        reduce_fn = {
            "sum": np.sum,
            "mean": np.mean
        }.get(self._reduce)
        self._batch_size, *_ = input_batch.shape
        return reduce_fn(input_batch, axis=0)

    @property
    def reduction_scaling(self) -> float:
        return {
            "sum": 1., "mean": 1. / self._batch_size
        }.get(self._reduce)

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplemented("Override this method!")

    def backward(self) -> np.ndarray:
        raise NotImplemented("Override this method!")