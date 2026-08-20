import numpy as np
# from numpy.typing import ArrayLike

class Estimator:
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()




# class Predictor(Estimator):
#     def predict(self, x: ArrayLike) -> np.ndarray:
#         raise NotImplementedError

#     def prd(self, x: ArrayLike) -> np.ndarray:
#         return self.predict(x)


# class Transformer(Estimator):
#     def transform(self, x: ArrayLike) -> np.ndarray:
#         raise NotImplementedError

#     def trf(self, x: ArrayLike) -> np.ndarray:
#         return self.transform(x)


