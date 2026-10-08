import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator


# TODO add to GradientDescent
# patience = 5,
# train_val_split = 0.2,
# verbose = True
class GradientDescent(Optimizer):
    def __init__(
            self,
            batch_size: int | None,
            iterations: int,
            tolerance: float,
            learning_rate: float,
        ):
        self.batch_size = batch_size
        self.iterations = iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate

        self.losses: NDArray = None
    
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
            if self.converged(grad) == True:
                break

        self.losses = losses
        return params

    def converged(self, grad: NDArray) -> bool:
        return np.linalg.norm(grad, ord=2) <= self.tolerance
