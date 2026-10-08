import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.base import Estimator
from pyml.base import Neighbor

from pyml.config.constants import EPS


class KNearestMean(Estimator, Neighbor):
    def __init__(
            self, 
            k: int = 5,
            chunksize: int = 100,
            weighted: bool = False,
        ):
        self.k = k
        self.chunksize = chunksize
        self.weighted = weighted

    def fit(self, x: ArrayLike, y: ArrayLike):
        self.x = np.asarray(x)  # [N, F]
        self.y = np.asarray(y)  # [N]

    def predict(self, x: ArrayLike) -> NDArray:
        x = np.asarray(x)
        index, distance = self.neighbors(x)  # [M, K], [M, K]
        y = self.y[index]  # [M, K]

        if self.weighted == True:
            weights = 1.0 / np.maximum(distance, EPS)
            return np.sum(weights * y, axis=1) / np.sum(weights, axis=1)
        
        return np.mean(y, axis=1)
