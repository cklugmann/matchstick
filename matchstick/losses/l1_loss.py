import numpy as np

from matchstick.losses.loss import Loss


class L1Loss(Loss):

    def __init__(self):
        super().__init__()
        self.cache = dict()

    def forward(self, pred: np.ndarray, target: np.ndarray):
        self.cache["target"] = target.copy()
        self.cache["pred"] = pred.copy()
        out = np.abs(pred.reshape(-1,) - target.reshape(-1,))
        return self.reduce_function(out)

    def backward(self) -> np.ndarray:

        pred = self.cache["pred"]
        target = self.cache["target"]

        dout = self.reduction_scaling * np.sign(pred - target)

        self.clear()

        return dout
