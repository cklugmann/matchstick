from typing import Optional
import numpy as np

from matchstick.losses.loss import Loss


class CrossEntropyFromLogits(Loss):

    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.cache = dict()

    def _to_one_hot(self, target: np.ndarray) -> np.ndarray:
        if np.issubdtype(target.dtype, np.integer):
            if self.num_classes is None:
                raise ValueError("You need to specify the number of classes for one-hot conversion.")
            classes = np.arange(self.num_classes)
            target = target.astype(classes.dtype)
            target = (
                    target.reshape(-1, 1) == classes.reshape(1, -1)
            )
        return target.astype(np.float32)

    def forward(self, pred: np.ndarray, target: np.ndarray):

        target = self._to_one_hot(target)
        pred = pred.astype(target.dtype)

        self.cache["target"] = target.copy()
        self.cache["pred"] = pred.copy()

        log_sum = np.log(np.sum(
            np.exp(pred), axis=-1, keepdims=True
        ))
        log_pred = pred - log_sum
        out = -np.sum(target * log_pred, axis=-1)

        return self.reduce_function(out)

    def backward(self) -> np.ndarray:

        pred = self.cache["pred"]
        target = self.cache["target"]

        # Goal: produce tensor of shape (M, num_classes)

        dlog = -1. * (target - np.exp(pred) / np.sum(np.exp(pred), axis=-1, keepdims=True))
        dout = self.reduction_scaling * dlog

        self.clear()

        return dout
