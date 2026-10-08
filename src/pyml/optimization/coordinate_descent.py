import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator


class CoordinateDescent(Optimizer):
    def __init__(
            self,
            iterations_max: int,
        ):
        self.iterations_max = iterations_max

    def run(
            self,
            estimator: Estimator,
            x: NDArray,
            y: NDArray,
            params: NDArray,
        ) -> tuple[NDArray, NDArray]:

        N, F = x.shape
        losses = np.full((self.iterations_max,), fill_value=np.nan)

        for i in range(self.iterations_max):
            for j in range(F):
                params[j] = estimator.coordinate(x, y, params, j)
            loss = estimator.loss(x, y, params)
            losses[i] = loss

        self.losses = losses
        return params
