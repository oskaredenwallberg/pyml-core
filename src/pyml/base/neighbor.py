import numpy as np
from numpy.typing import ArrayLike, NDArray


class Neighbor:
    k: int
    chunksize: int

    def fit(self, x: ArrayLike, y: ArrayLike):
        raise NotImplementedError

    def predict(self, x: ArrayLike) -> NDArray:
        raise NotImplementedError

    def prd(self, x: ArrayLike) -> NDArray:
        return self.predict(x)

    def neighbors(self, x: ArrayLike) -> tuple[NDArray, NDArray]:
        M, F = x.shape

        b = self.x
        bb = np.sum(b**2, axis=1)[None,:]
        index = np.empty((M, self.k), dtype=int)

        rows = np.arange(self.chunksize)[:, None]
        distance = np.empty((M, self.k), dtype=float)
        
        for i0 in range(0, M, self.chunksize):
            i1 = i0 + min(M-i0, self.chunksize)
            a = x[i0:i1]
            aa = np.sum(a**2, axis=1)[:,None]
            ab = a @ b.T
            norm2sq = aa + bb - 2 * ab
            argpart = np.argpartition(norm2sq, self.k-1, axis=1)[:, :self.k]

            index[i0:i1] = argpart
            distance[i0:i1] = norm2sq[rows[:i1-i0], argpart] ** 0.5

        return index, distance
