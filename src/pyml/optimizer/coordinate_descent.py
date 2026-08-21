import numpy as np
from numpy.typing import NDArray

from pyml.optimizer.base import Optimizer
from pyml.estimator.base import Estimator


class CoordinateDescent(Optimizer):
    def __init__(
            self,
            iterations: int,
        ):
        self.iterations = iterations

    def run(
            self,
            estimator: Estimator,
            x: NDArray,
            y: NDArray,
            params: NDArray,
        ) -> tuple[NDArray, NDArray]:

        N, F = x.shape
        losses = np.full((self.iterations,), fill_value=np.nan)

        for i in range(self.iterations):
            for j in range(F):
                params[j] = estimator.coordinate(x, y, params, j)
            loss = estimator.loss(x, y, params)
            losses[i] = loss

        return losses, params

