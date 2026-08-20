import numpy as np
from typing import Callable
from numpy.typing import NDArray

from pyml.optimizer.base import Optimizer

# TODO add to GradientDescent
# patience = 5,
# train_val_split = 0.2,
# verbose = True

class GradientDescent:
    def __init__(
            self,
            batch_size: int | None,
            iterations: int,
            learning_rate: int,
        ):
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def run(
            self,
            objective: Callable,
            x: NDArray, 
            y: NDArray, 
            params: NDArray,
        ) -> tuple[NDArray, NDArray]:
        
        losses = np.full((self.iterations,), fill_value=np.nan)
        N, F = x.shape
        x_batch, y_batch = x.copy(), y.copy()
        index = np.arange(N)

        for i in range(self.iterations):
            if self.batch_size is not None:
                np.random.shuffle(index)
                index_batch = index[:self.batch_size]
                x_batch = x[index_batch]
                y_batch = y[index_batch]
            loss, grad = objective(x_batch, y_batch, params)
            losses[i] = loss
            params -= self.learning_rate * grad

        return losses, params

