import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from matchstick.layers import Linear
from matchstick.models import Sequential
from matchstick.losses import CrossEntropyFromLogits
from matchstick.losses.regularizer import L1
from matchstick.optimizers import Adam


def target_function(X: np.ndarray) -> np.ndarray:
    x, y = X.T
    target = (np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) < 1 / 4).astype(int)
    return target


def get_batch(num_points: int) -> [np.ndarray, np.ndarray]:
    x = np.random.uniform(low=0, high=1, size=(num_points,))
    y = np.random.uniform(low=0, high=1, size=(num_points,))
    X = np.concatenate([
        x.reshape(-1, 1),
        y.reshape(-1, 1)
    ], axis=-1)
    target = target_function(X)
    return X, target


def main():

    np.random.seed(42)
    save_folder = "static"

    model = Sequential(
        [
            Linear(out_features=256, regularizer={"weights": L1(scale=5e-4)}),
            Linear(
                out_features=2,
                activation=False,
            ),
        ]
    )
    loss = CrossEntropyFromLogits(num_classes=2)
    model.compile(loss=loss, in_features=2)
    params = model.params
    optimizer = Adam(params=params, learning_rate=1e-2)

    def predict(xs: np.ndarray) -> np.ndarray:
        logits = model(xs)
        pred_label = np.argmax(logits, axis=-1)
        return pred_label

    num_steps = 128

    points_per_dim = 128
    dom = np.linspace(0, 1, num=points_per_dim)
    XX, YY = np.meshgrid(dom, dom)
    xx, yy = map(
        lambda t: t.reshape(-1, 1),
        [XX, YY]
    )
    X_test = np.concatenate([
        xx, yy
    ], axis=-1)
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
    fig.savefig(os.path.join(save_folder, "classification_loss.png"), bbox_inches="tight")
    plt.show()

    pred_test = predict(X_test)
    im = (255 * pred_test.reshape(points_per_dim, points_per_dim)).astype(np.uint8)
    im = np.flipud(im)
    im = np.tile(np.expand_dims(im, -1), reps=(1, 1, 3))
    im = Image.fromarray(im)

    coord0 = int(points_per_dim * 0.25)
    coord1 = int(points_per_dim * 0.75)

    two_point_list = [
        (coord0, coord0),
        (coord1, coord1)
    ]
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy=two_point_list, outline="red")

    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.set_xticks([0, points_per_dim-1])
    ax.set_yticks([0, points_per_dim-1])
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([1, 0])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Model prediction for $x^2 + y^2 < 1/4$")
    fig.savefig(os.path.join(save_folder, "classification_prediction.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
