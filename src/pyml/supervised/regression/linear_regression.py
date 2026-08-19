import numpy as np
from typing import Literal

from ..base.linear import LinearModel
from ..base.linear import lasso, ridge, elastic_net
from ...utils.math.activations import linear
from ...utils.metrics.regression import mse_score, mae_score

# --------------------- LINEAR REGRESSION MODELS ---------------------
# - LinearRegression        (LinearModel)
# - LinearLASSO             (LinearModel)
# - LinearRidge             (LinearModel)
# - LinearElasticNet        (LinearModel)

class LinearRegression(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self._activation = linear
        self.n_features = None

    def compute_loss(self, y_true, y_pred) -> float:
        """
        Includes no penalty in normal LinearRegression
        """
        loss = mse_score(y_true, y_pred)
        return loss

    def gradient_update(self, x, y_true, y_pred) -> None:
        """
        Includes no penalty in normal LinearRegression.

        >>> loss(X, y) = 1/m sum((y - xw - b)^2)
        >>> # Linear Regression partial differentials
        >>> d/dw loss(w, b) = -2/m * x(y - xw - b)
        >>> d/db loss(w, b) =   -2(y - xw - b) 
        """
        error = y_true - y_pred  # y - (xw + b)
        m = x.shape[0]
        gradient_w = -2 * x.T @ error / m
        gradient_b = -2 * error.mean()

        self.w -= self.learning_rate * gradient_w
        self.b -= self.learning_rate * gradient_b

    def fit_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ordinary Least Squares Linear Regression implementaion.

        X is the design matrix, B the coefficient vector and y the obervation
        vector.

        >>> XB = y
        >>> X^T @ XB = X^T @ y
        >>> B = (X^T @ X)^-1 @ X^T y  # !not solvable for singular matrix, IMT
        """ 
        n_samples = X.shape[0]
        ones = np.ones(n_samples).reshape(-1, 1)
        X = np.c_[ones, X]  # prepend ones for intercept
        parameters = np.linalg.inv(X.T @ X) @ (X.T @ y)
        parameters = parameters.ravel()
        self.w = parameters[1:]
        self.b = parameters[0]



class LinearLASSO(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            alpha : float = 1,  # regularisation strength
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self._activation = linear
        self.alpha = alpha
        self.n_features = None

    def compute_loss(self, y_true, y_pred) -> float:
        """
        ...
        """
        penalty = lasso(self.w)  # l1 norm
        loss = mse_score(y_true, y_pred) + self.alpha * penalty
        return loss

    def gradient_update(self, x, y_true, y_pred) -> None:
        """
        Compute the LASSO gradient with L1 norm as regularisation.

        >>> loss(X, y) = sum((y - xw - b)^2) + alpha * sum(|w|)
        >>> # LASSO partial differentials
        >>> d/dw loss(w, b) = -2*x(y - xw - b) + alpha * sign(w)
        >>> d/db loss(w, b) =   -2(y - xw - b) 
        """
        error = y_true - y_pred  # y - (xw + b)
        m = x.shape[0]
        gradient_w = -2 * x.T @ error / m + self.alpha * np.sign(self.w)
        gradient_b = -2 * error.mean()

        self.w -= self.learning_rate * gradient_w
        self.b -= self.learning_rate * gradient_b



class LinearRidge(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            lamda : float = 1,  # regularisation strength
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self._activation = linear
        self.lamda = lamda
        self.n_features = None

    def compute_loss(self, y_true, y_pred) -> float:
        """
        ...
        """
        penalty = ridge(self.w)  # l2 norm
        loss = mse_score(y_true, y_pred) + self.lamda * penalty
        return loss

    def gradient_update(self, x, y_true, y_pred) -> None:
        """
        Compute the Ridge gradient with L2 norm as regularisation.

        >>> loss(X, y) = sum((y - xw - b)^2) + lamda * sum(w^2)
        >>> # Ridge partial differentials
        >>> d/dw loss(w, b) = -2*x(y - xw - b) + 2 * lamda * w
        >>> d/db loss(w, b) =   -2(y - xw - b) 
        """
        error = y_true - y_pred
        m = x.shape[0]
        
        gradient_w = -2 * x.T @ error / m + 2 * self.lamda * self.w
        gradient_b = -2 * error.mean()

        self.w -= self.learning_rate * gradient_w
        self.b -= self.learning_rate * gradient_b



class LinearElasticNet(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            alpha : float = 1,  # regularisation strength
            l1_ratio : float = 0.5,  # l1 norm proportion of penalty
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self._activation = linear
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_features = None

    def compute_loss(self, y_true, y_pred) -> float:
        """
        ...
        """
        penalty = elastic_net(self.w, self.l1_ratio)  # weighted sum of l1 norm and l2 norm
        loss = mse_score(y_true, y_pred) + self.alpha * penalty
        return loss

    def gradient_update(self, x, y_true, y_pred) -> None:
        """
        Compute the Elastic Net gradient with L1 and L2 norm as regularisation.

        >>> penalty(w) = l1_ratio * sum(|w|) + (1 - l1_ratio) * sum(w^2)
        >>> loss(X, y) = sum((y - xw - b)^2) + alpha * penalty
        >>> # Elastic net partial differentials
        >>> d/dw penalty(w) = l1_ratio * sign(w) + (1 - l1_ratio) * 2 * w
        >>> d/dw loss(w, b) = -2*x(y - xw - b) + alpha * d/dw penalty(w)
        >>> d/db loss(w, b) =   -2(y - xw - b)
        """
        error = y_true - y_pred
        m = x.shape[0]

        gradient_penalty = self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * 2 * self.w
        gradient_w = -2 * x.T @ error / m + self.alpha * gradient_penalty
        gradient_b = -2 * error.mean()

        self.w -= self.learning_rate * gradient_w
        self.b -= self.learning_rate * gradient_b
