import numpy as np
from numpy.typing import ArrayLike, NDArray

class Linear:
    params: NDArray | None

    def predict(self, x: ArrayLike) -> NDArray:
        assert self.params is not None
        x = np.asarray(x).copy()
        N = x.shape[0]
        x = np.c_[np.ones((N, 1)), x]

        return x @ self.params

    def prd(self, x: ArrayLike) -> NDArray:
        return self.predict(x)

    @property
    def coefficients(self) -> NDArray:
        assert self.params is not None
        return self.params[1:]

    @property
    def intercept(self) -> NDArray:
        assert self.params is not None
        return self.params[0]
