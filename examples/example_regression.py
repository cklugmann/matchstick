import os

import numpy as np
import matplotlib.pyplot as plt

from matchstick.layers import Linear
from matchstick.models import Sequential
from matchstick.losses import L1Loss
from matchstick.losses.regularizer import L1
from matchstick.optimizers import Adam


def target_function(x: np.ndarray) -> np.ndarray:
    return np.exp(-x ** 2 / 0.02)


def get_batch(num_points: int) -> [np.ndarray, np.ndarray]:
    x = np.random.uniform(low=0, high=1, size=(num_points, 1))
    y = target_function(x)
    return x, y


def main():

    np.random.seed(42)
    save_folder = "static"

    layers = [
        Linear(out_features=64, regularizer={"weights": L1(scale=5e-4)}),
        Linear(out_features=64, regularizer={"weights": L1(scale=5e-4)}),
        Linear(
            out_features=1,
            activation=False
        ),
    ]

    model = Sequential(layers)
    loss = L1Loss()
    model.compile(loss=loss, in_features=1)
    params = model.params
    optimizer = Adam(params=params, learning_rate=4e-3)

    num_steps = 60

    x_test = np.linspace(start=0, stop=1, num=512).reshape(-1, 1)
    y_test = target_function(x_test)
    train_data = [get_batch(num_points=64) for _ in range(num_steps)]

    losses = {
        "main": list(),
        "regularization": list(),
        "total": list()
    }

    for step, (xs, targets) in enumerate(train_data):

        _ = model(xs)
        loss_dict = model.compute_loss(targets)
        grads = model.grads
        optimizer.apply(grads)

        for loss_type, loss_val in loss_dict.items():
            losses[loss_type].append(loss_val)

    y_pred = model(x_test)
    x_test, y_test, y_pred = map(
        lambda t: t.reshape(-1,),
        [x_test, y_test, y_pred]
    )

    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, label="true")
    ax.plot(x_test, y_pred, label="pred")
    ax.grid(True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Model prediction vs target function $f(x) = \\exp(-x^2/0.02)$")
    fig.savefig(os.path.join(save_folder, "regression_prediction.png"), bbox_inches="tight")
    plt.legend()

    plt.show()

    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    axs = axs.reshape(-1,)
    for ax, kv in zip(axs, losses.items()):
        loss_type, loss_values = kv
        steps = np.arange(len(loss_values))
        ax.plot(steps, loss_values)
        ax.grid(True)
        ax.set_xlabel("steps")
        ax.set_ylabel("loss")
        ax.set_title(loss_type)
    fig.savefig(os.path.join(save_folder, "regression_loss.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
