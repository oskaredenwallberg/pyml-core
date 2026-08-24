import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.estimator.base import Estimator
from pyml.estimator.base import Neighbor

from pyml.config.constants import EPS


class KNearestVoting(Estimator, Neighbor):
    def __init__(
            self, 
            k: int = 5,
            chunksize: int = 100,
            weighted: bool = False,
        ):
        self.k = k
        self.chunksize = chunksize
        self.weighted = weighted
        self.classes = None
        self.inverse = None

    def fit(self, x: ArrayLike, y: ArrayLike):
        self.x = np.asarray(x)
        classes, inverse = np.unique(y, return_inverse=True)
        self.classes = classes
        self.inverse = inverse

    def predict(self, x: ArrayLike) -> NDArray:
        x = np.asarray(x)
        M, F = x.shape
        index, distance = self.neighbors(x)

        C = self.classes.size
        classindex = self.inverse[index]
        predictions = np.empty((M,), dtype=self.classes.dtype)
        
        weights = 1.0 / np.maximum(distance, EPS) if self.weighted == True else None

        for i0 in range(0, M, self.chunksize):
            i1 = i0 + min(self.chunksize, M-i0)
            B = i1 - i0
            
            r = np.ravel(classindex[i0:i1] + C * np.arange(B)[:,None])
            w = np.ravel(weights[i0:i1]) if weights is not None else None
            counts = np.bincount(r, weights=w, minlength=B*C).reshape(B, C)

            predictions[i0:i1] = self.classes[np.argmax(counts, axis=1)]

        return predictions
