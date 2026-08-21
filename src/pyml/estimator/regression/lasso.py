import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.estimator.base import Estimator
from pyml.estimator.base.linear import Linear
from pyml.optimizer import CoordinateDescent


class LassoCD(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            optimizer: CoordinateDescent
            ):
        self.lamda = lamda
        self.optimizer = optimizer
        self.params: NDArray = None
        self.losses: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        params = np.zeros(F+1)
        losses, params = self.optimizer.run(self, x, y, params)

        self.losses = losses
        self.params = params

    def coordinate(self, x: NDArray, y: NDArray, theta: NDArray, j: int):
        N, F = x.shape
        xj = x[:, j]
        rj = y - x @ theta + xj * theta[j]
        xr = xj @ rj
        xx = xj @ xj

        if j == 0:
            return xr / xx
        return soft_threshold(xr, N * self.lamda / 2) / xx

    def loss(self, x: NDArray, y: NDArray, theta: NDArray):
        loss = np.mean((y - x @ theta) ** 2) 
        loss += self.lamda * np.sum(np.abs(theta[1:]))
        return loss


def soft_threshold(a, b):
    if a > b:
        return a - b
    if a < -b:
        return a + b
    return 0.0




# L(theta) = 1/N * l2(y - X @ theta)^2 + lamda * l1(theta)
# L(theta) = 1/N * Sum( (y - X @ theta)^2 ) + lamda * Sum( abs(theta[1:]) )
