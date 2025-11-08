#!/usr/bin/env python
"""Python code submission file.
IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import json
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(10)


class NeuralNetwork:
    def __init__(self, n_in, n_hidden, n_out):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.theta = self.init_params()

    def init_params(self):
        theta = {}
        theta["W0"] = np.array(0)
        theta["b0"] = np.array(0)
        theta["W1"] = np.array(0)
        theta["b1"] = np.array(0)
        return theta

    def export_model(self):
        with open(f"model.json", "w") as fp:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, fp)


def task():
    """Neural Network

    Requirements for the plots:
        - ax[0] Plot showing the training loss and training accuracy
        - ax[1] Plot showing the confusion matrix on the test data (using matplotlib.pyplot.imshow)
        - ax[2] Scatter plot showing the labeled training data
        - ax[3] Plot showing the learned decision boundary weighted by the logits output (using matplotlib.pyplot.imshow)
    """
    with np.load("data.npz") as data_set:
        # get the training data
        x_train = data_set["x_train"]
        y_train = data_set["y_train"]

        # get the test data
        x_test = data_set["x_test"]
        y_test = data_set["y_test"]

    print(f"\nTraining set with {x_train.shape[0]} data samples")
    print(f"Test set with {x_test.shape[0]} data samples")

    extent = (
        x_train[:, 0].min(),
        x_train[:, 0].max(),
        x_train[:, 1].min(),
        x_train[:, 1].max(),
    )
    cl_colors = ["blue", "orange"]
    cmap = colors.ListedColormap(cl_colors)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    # ax[0] Plot showing the training loss and training accuracy
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("Epochs")

    # ax[1] Plot showing the confusion matrix on the test data (using matplotlib.pyplot.imshow)
    conf_mat = np.eye(len(np.unique(y_train)))
    conf = ax[1].imshow(conf_mat), ax[1].set_title("Confusion matrix (test data)")
    fig.colorbar(conf[0], ax=ax[1], shrink=0.5)
    (
        ax[1].set_xticks(list(np.arange(len(np.unique(y_train))))),
        ax[1].set_xlabel("predicted label"),
    )
    (
        ax[1].set_yticks(list(np.arange(len(np.unique(y_train))))),
        ax[1].set_ylabel("actual label"),
    )

    # ax[2] Scatter plot showing the labeled training data
    for idx, cl in enumerate(["class 1", "class 2"]):
        ax[2].scatter(
            x_train[:, 0][y_train == idx],
            x_train[:, 1][y_train == idx],
            label=cl,
            c=cl_colors[idx],
        )
    ax[2].set_title("Training data")
    ax[2].legend()
    ax[2].set_xlabel(r"$x_1$"), ax[2].set_ylabel(r"$x_2$")

    # ax[3] Plot showing the learned decision boundary weighted by the logits output (using matplotlib.pyplot.imshow)
    N = 500
    ax[3].imshow(
        np.ones((N, N)),
        alpha=np.random.rand(N, N),
        origin="lower",
        extent=extent,
        cmap=cmap,
    )
    ax[3].set_title("Learned decision boundary")
    ax[3].set_xlabel(r"$x_1$"), ax[3].set_xlabel(r"$x_1$")

    """
    Start your code here

    """

    return fig


if __name__ == "__main__":
    pdf = PdfPages("nn_figures.pdf")
    fig = task()
    pdf.savefig(fig)
    pdf.close()
