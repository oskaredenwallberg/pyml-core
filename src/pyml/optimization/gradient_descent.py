import numpy as np
from numpy.typing import NDArray

# from pyml.base import Estimator
from pyml.base import Optimizer


# TODO add to GradientDescent
# patience = 5,
# train_val_split = 0.2,
# verbose = True
class GradientDescent(Optimizer):
    def __init__(
            self,
            batch_size: int | None,
            iterations_max: int,
            tolerance: float,
            learning_rate: float,
        ):
        self.batch_size = batch_size
        self.iterations_max = iterations_max
        self.tolerance = tolerance
        self.learning_rate = learning_rate

        self.losses: NDArray = None
    
    def run(self, estimator, x,  y,  params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        N, F = x.shape
        x_batch, y_batch = x, y
        index = np.arange(N)

        for i in range(self.iterations_max):
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
