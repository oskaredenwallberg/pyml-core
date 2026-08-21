from numpy.typing import NDArray

from pyml.estimator.base import Estimator

class Optimizer:
    def run(
            self,
            estimator: Estimator,
            x: NDArray,
            y: NDArray,
            params: NDArray
        ) -> tuple[NDArray, NDArray]:
        raise NotImplementedError
