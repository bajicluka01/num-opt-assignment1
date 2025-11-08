#!/usr/bin/env python
"""Python code submission file.
IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import approx_fprime
from typing import Callable

# Modify the following global variables to be used in your functions
""" Start of your code
"""
alpha = -5
beta = 3
a = [-1, 3]
d = 2.5
b = None
D = None
A = None
""" End of your code
"""


def task1():
    """Characterization of Functions
    Requirements for the plots:
        - ax[0, 0] Contour plot for a)
        - ax[0, 1] Contour plot for b)
        - ax[1, 0] Contour plot for c)
        - ax[1, 1] Contour plot for d)
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Task 1 - Contour plots of functions", fontsize=16)
    ax[0, 0].set_title("a)")
    ax[0, 0].set_xlabel("$x_1$")
    ax[0, 0].set_ylabel("$x_2$")
    ax[0, 1].set_title("b)")
    ax[0, 1].set_xlabel("$x_1$")
    ax[0, 1].set_ylabel("$x_2$")
    ax[1, 0].set_title("c)")
    ax[1, 0].set_xlabel("$x_1$")
    ax[1, 0].set_ylabel("$x_2$")
    ax[1, 1].set_title("d)")
    ax[1, 1].set_xlabel("$x_1$")
    ax[1, 1].set_ylabel("$x_2$")
    """ Start of your code
    """
    x1, x2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
    #ax[0, 0].contour(x1, x2, x1**2 + 0.5 * x2**2, 50)

    ax[0, 0].contour(x1, x2, func_1a([x1,x2]), 50)
    ax[0, 1].contour(x1, x2, func_1b([x1,x2]), 50)
    ax[1, 0].contour(x1, x2, func_1c([x1,x2]), 50)
    ax[1, 1].contour(x1, x2, func_1d([x1,x2]), 50)

    """ End of your code
    """
    return fig


# Modify the function bodies below to be used for function value and gradient computation


def approx_grad_task1(
    func: Callable[[np.ndarray], float], x: np.ndarray, eps: float
) -> np.ndarray:
    """Numerical Gradient Computation
    @param func function that takes a vector
    @param x Vector of size (2,)
    @param eps small value for numerical gradient computation
    This function shall compute the gradient approximation for a given point 'x' and a function 'func'
    using the given central differences formulation for 2D functions. (Task1 functions)
    @return The gradient approximation
    """
    assert len(x) == 2
    pass


def approx_grad_task2(
    func: Callable[[np.ndarray], float], x: np.ndarray, eps: float
) -> np.ndarray:
    """Numerical Gradient Computation
    @param func function that takes a vector
    @param x Vector of size (n,)
    @param eps small value for numerical gradient computation
    This function shall compute the gradient approximation for a given point 'x' and a function 'func'
    using scipy.optimize.approx_fprime(). (Task2 functions)
    @return The gradient approximation
    """
    pass


def func_1a(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1a) at a given point x
    @param x Vector of size (2,)
    """

    # computes a dot product between two vectors (given as lists, no need for transpose)
    def dot_prod(v1, v2):
        out = 0
        for v1i, v2i in zip(v1, v2):
            out += v1i*v2i
        return out
    
    return pow(dot_prod(a, x)-d, 2)


def grad_1a(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1a) at a given point x
    @param x Vector of size (2,)
    """
    return [2*x[0] - 6*x[1] + 5, -6*x[0] + 18*x[1] - 15]


def func_1b(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1b) at a given point x
    @param x Vector of size (2,)
    """
    
    return pow(x[0]-2, 2)+x[0]*x[1]*x[1]-2


def grad_1b(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1b) at a given point x
    @param x Vector of size (2,)
    """
    
    return [2*x[0] + x[1]*x[1] - 4, 2*x[0]*x[1]]


def func_1c(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1c) at a given point x
    @param x Vector of size (2,)
    """
    # computes the l2 norm of the given vector
    def l2_norm(v):
        return np.sqrt(sum(vi*vi for vi in v))
    
    tmp = pow(l2_norm(x), 2)
    return x[0]*x[0] + x[0] * tmp + tmp

def grad_1c(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1c) at a given point x
    @param x Vector of size (2,)
    """
    
    return [3*x[0]*x[0] + 4*x[1] + x[1]*x[1], 2*x[1] + 2*x[0]*x[1]]


def func_1d(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """
    
    return alpha*x[0]*x[0] - 2*x[0] + beta*x[1]*x[1]


def grad_1d(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """
    
    return [2*alpha*x[0] - 2, 2*beta*x[1]]


def func_2a(x: np.ndarray) -> float:
    """Computes and returns the function value for function 2a) at a given point x
    @param x Vector of size (n,)
    """
    pass


def grad_2a(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 2a) at a given point x
    @param x Vector of size (n,)
    """
    pass


def func_2b(x: np.ndarray) -> float:
    """Computes and returns the function value for function 2b) at a given point x
    @param x Vector of size (n,)
    """
    pass


def grad_2b(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 2b) at a given point x
    @param x Vector of size (n,)
    """
    pass


def func_2c(x: np.ndarray) -> float:
    """Computes and returns the function value for function 2c) at a given point x
    @param x Vector of size (n,)
    """
    pass


def grad_2c(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 2c) at a given point x
    @param x Vector of size (n,)
    """
    pass


def task3():
    """Numerical Gradient Verification
    ax[0] to ax[3] Bar plot comparison, analytical vs numerical gradient for Task 1
    ax[4] to ax[6] Bar plot comparison, analytical vs numerical gradient for Task 2

    """
    fig, ax = plt.subplot_mosaic(
        [
            3 * ["1a)"] + 3 * ["1b)"] + 3 * ["1c)"] + 3 * ["1d)"],
            4 * ["2a)"] + 4 * ["2b)"] + 4 * ["2c)"],
        ],
        figsize=(15, 10),
        constrained_layout=True,
    )
    fig.suptitle("Task 3 - Numerical vs analytical", fontsize=16)
    keys = ["1a)", "1b)", "1c)", "1d)", "2a)", "2b)", "2c)"]
    for k in keys:
        ax[k].set_title(k)
        ax[k].set_xlabel(r"$\epsilon$")
        ax[k].set_ylabel("Error")

    """ Start of your code
    """
    eps = np.logspace(-15, 2, 100)

    """ End of your code
    """
    return fig


if __name__ == "__main__":
    tasks = [task1, task3]

    pdf = PdfPages("optimality_figures.pdf")
    for task in tasks:
        retval = task()
        pdf.savefig(retval)
    pdf.close()
