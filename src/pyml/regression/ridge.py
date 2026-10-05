import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.base import Estimator, Optimizer
from pyml.base import Linear


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

    def predict(self, x: ArrayLike) -> NDArray:
        return self.linear(x)


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

    def predict(self, x: ArrayLike) -> NDArray:
        return self.linear(x)


class RidgeGD(Estimator, Linear):
    def __init__(
            self,
            lamda: float,  # regularization strength
            batch_size: int | None = None,
            iterations: int = 100,
            learning_rate: float = 1e-3,
        ):
        self.lamda = lamda
        self.optimizer = GradientDescent(batch_size, iterations, learning_rate)
        self.params: NDArray = None
        self.losses: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        params = np.zeros(F+1)
        losses, params = self.optimizer.run(self, x, y, params)

        self.losses = losses
        self.params = params

    def gradient(self, x: NDArray, y: NDArray, theta: NDArray) -> NDArray:
        N, F = x.shape
        grad = -2/N * x.T @ (y - x @ theta)
        grad[1:] += 2 * self.lamda * theta[1:]
        return grad

    def loss(self, x: NDArray, y: NDArray, theta: NDArray) -> NDArray:
        loss = np.mean((y - x @ theta) ** 2)
        loss += self.lamda * np.sum(theta[1:] ** 2)
        return loss

    def predict(self, x: ArrayLike) -> NDArray:
        return self.linear(x)


class GradientDescent(Optimizer):
    def __init__(
            self,
            batch_size: int | None,
            iterations: int,
            learning_rate: float,
        ):
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def run(
            self,
            estimator: Estimator,
            x: NDArray, 
            y: NDArray, 
            params: NDArray,
        ) -> tuple[NDArray, NDArray]:
        
        losses = np.full((self.iterations,), fill_value=np.nan)
        N, F = x.shape
        x_batch, y_batch = x, y
        index = np.arange(N)

        for i in range(self.iterations):
            if self.batch_size is not None:
                np.random.shuffle(index)
                index_batch = index[:self.batch_size]
                x_batch = x[index_batch]
                y_batch = y[index_batch]
            grad = estimator.gradient(x_batch, y_batch, params)
            params -= self.learning_rate * grad
            loss = estimator.loss(x_batch, y_batch, params)
            losses[i] = loss

        return losses, params

# TODO add to GradientDescent
# patience = 5,
# train_val_split = 0.2,
# verbose = True


class ConjugateDescent(Optimizer):
    pass # TODO
