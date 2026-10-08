import numpy as np

class Estimator:
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()
