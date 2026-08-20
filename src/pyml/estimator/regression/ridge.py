import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.estimator.base import Estimator
from pyml.estimator.base.linear import Linear
from pyml.optimizer import GradientDescent


class RidgeGD(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            optimizer: GradientDescent,
        ):
        self.lamda = lamda
        self.optimizer = optimizer
        self.params: NDArray = None
        self.losses: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        params = np.zeros(F+1)
        losses, params = self.optimizer.run(self._objective, x, y, params)

        self.losses = losses
        self.params = params
    
    def _objective(self, x: NDArray, y: NDArray, theta: NDArray):
        N, F = x.shape
        penalty = theta.copy()
        penalty[0] = 0.0
        error = y - x @ theta
        loss = np.mean(error ** 2) + self.lamda * np.sum(penalty ** 2)
        grad = -2/N * x.T @ error + self.lamda * 2 * penalty
        return loss, grad


class RidgeQR(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            ):
        self.lamda = lamda
        self.params: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        eye = np.eye(F+1)
        eye[0,0] = 0.0
        x_stacked = np.r_[x, np.sqrt(N * self.lamda) * eye]
        y_stacked = np.r_[y, np.zeros(F+1)]
        q, r = np.linalg.qr(x_stacked)
        params = np.linalg.solve(r, q.T @ y_stacked)

        self.params = params


class RidgeCholesky(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            ):
        self.lamda = lamda
        self.params: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        eye = np.eye(F+1)
        eye[0,0] = 0.0
        a = x.T @ x + N * self.lamda * eye
        b = x.T @ y
        l = np.linalg.cholesky(a)
        z = np.linalg.solve(l, b)
        params = np.linalg.solve(l.T, z)

        self.params = params
