import numpy as np
from abc import ABC, abstractmethod

from ...utils.math.distance import euclidean_distance

# --------------------- K NEAREST NEIGHBORS BASE CLASSES ---------------------
# KNNModel      (ABC)

class KNNModel(ABC):
    def __init__(
            self,
            k : int = 5,  # any odd value
        ):
        super().__init__()
        # Parameters
        self.k = k
        self.x_train = None
        self.y_train = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        KNN is parameters-less. 
        No formal training other than storing data for future reference during 
        majority voting and mean calculations. 

        Parameters
        ----------
        X : ndarray
            Array of input feature data. 
        y : ndarray
            Array of input target data.
        """
        self.x_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ...
        
        Parameters
        ---------
        X : ndarray
            Array of input data to be evaluated.

        Returns
        -------
        ndarray
            Array of target predictions.
        """ 
        m = X.shape[0]  # num inputted datapoints
        predictions = np.zeros(m)

        for i in range(m):
            x = X[i]
            distances = euclidean_distance(x, self.x_train, axis=1)
            predictions[i] = self.evaluate(distances)

        return predictions

    @abstractmethod
    def evaluate(self, distances: np.ndarray) -> int|float:
        pass

    @property
    def is_fitted(self) -> bool:
        """
        ...
        """
        return self.x_train is not None and self.y_train is not None
    



    # def k_nearest(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Find the indicies of the k training datapoints nearest to
    #     the inputted datapoint x. Nearest is defined by euclidean
    #     distance.

    #     Parameters
    #     ----------
    #     x : ndarray
    #         An array containing the feature values of a single datapoint.

    #     Returns
    #     -------
    #     ndarray
    #         An array of indicies of the k training datapoints nearest x.  
    #     """
        