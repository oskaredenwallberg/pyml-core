from typing import Callable
from numpy.typing import NDArray

class Optimizer:
    def run(
            self,
            objective: Callable,
            x: NDArray,
            y: NDArray,
            params: NDArray
        ) -> tuple[NDArray, NDArray]:
        raise NotImplementedError
