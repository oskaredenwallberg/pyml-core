import numpy as np
# from numpy.typing import ArrayLike

class Estimator:
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()
