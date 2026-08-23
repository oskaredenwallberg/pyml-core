import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.estimator.base import Estimator
from pyml.estimator.base import Neighbor


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

    def predict(self, x: ArrayLike) -> NDArray:
        x = np.asarray(x)
        index, distance = self.neighbors(x)
        y = self.y[index]

        if self.weighted == True:
            weight = 1.0 / distance
            return np.sum(weight * y, axis=1) / np.sum(weight, axis=1)
        
        return np.mean(y, axis=1)
