import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.estimator.base import Estimator
from pyml.estimator.base.linear import Linear
from pyml.optimizer.base import Optimizer


def objective(x: NDArray, y: NDArray, theta: NDArray):
    N = x.shape[0]
    error = y - x @ theta
    loss = np.mean(error ** 2)
    grad = -2/N * x.T @ error
    return loss, grad


class LinearRegression(Estimator, Linear):
    def __init__(
            self,
            optimizer: Optimizer,
            ):
        self.optimizer = optimizer
        self.params: NDArray = None
        self.losses: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        params = np.zeros(F+1)
        losses, params = self.optimizer.run(objective, x, y, params)

        self.losses = losses
        self.params = params
    
    def predict(self, x: ArrayLike) -> NDArray:
        assert self.params is not None, self.params
        x = np.asarray(x).copy()
        N = x.shape[0]
        x = np.c_[np.ones((N, 1)), x]

        return x @ self.params




def qr(x: NDArray, y: NDArray):
    q, r = np.linalg.qr(x)
    theta = np.linalg.solve(r, q.T @ y)
    return theta

def cholesky(x: NDArray, y: NDArray):
    a = x.T @ x
    b = x.T @ y
    l = np.linalg.cholesky(a)
    z = np.linalg.solve(l, b)
    theta = np.linalg.solve(l.T, z)
    return theta




# class LinearLASSO(LinearModel):
#     def __init__(
#             self,
#             learning_rate = 0.01,
#             alpha : float = 1,  # regularisation strength
#             epochs = 200,
#             batch_size = 32,
#             keep_rest = False,
#             training_method = 'mbsgd',
#             patience = 5,
#             train_val_split = 0.2,
#             verbose = True):
#         super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
#         self.alpha = alpha
#         self.n_features = None

#     def compute_loss(self, y_true, y_pred) -> float:
#         penalty = lasso(self.w)  # l1 norm
#         loss = mse_score(y_true, y_pred) + self.alpha * penalty
#         return loss

#     def gradient_update(self, x, y_true, y_pred) -> None:

#         error = y_true - y_pred  # y - (xw + b)
#         m = x.shape[0]
#         gradient_w = -2 * x.T @ error / m + self.alpha * np.sign(self.w)
#         gradient_b = -2 * error.mean()

#         self.w -= self.learning_rate * gradient_w
#         self.b -= self.learning_rate * gradient_b



# class LinearRidge(LinearModel):
#     def __init__(
#             self,
#             learning_rate = 0.01,
#             lamda : float = 1,  # regularisation strength
#             epochs = 200,
#             batch_size = 32,
#             keep_rest = False,
#             training_method = 'mbsgd',
#             patience = 5,
#             train_val_split = 0.2,
#             verbose = True):
#         super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
#         self.lamda = lamda
#         self.n_features = None

#     def compute_loss(self, y_true, y_pred) -> float:
#         penalty = ridge(self.w)  # l2 norm
#         loss = mse_score(y_true, y_pred) + self.lamda * penalty
#         return loss

#     def gradient_update(self, x, y_true, y_pred) -> None:

#         error = y_true - y_pred
#         m = x.shape[0]
        
#         gradient_w = -2 * x.T @ error / m + 2 * self.lamda * self.w
#         gradient_b = -2 * error.mean()

#         self.w -= self.learning_rate * gradient_w
#         self.b -= self.learning_rate * gradient_b


# class LinearElasticNet(LinearModel):
#     def __init__(
#             self,
#             learning_rate = 0.01,
#             alpha : float = 1,  # regularisation strength
#             l1_ratio : float = 0.5,  # l1 norm proportion of penalty
#             epochs = 200,
#             batch_size = 32,
#             keep_rest = False,
#             training_method = 'mbsgd',
#             patience = 5,
#             train_val_split = 0.2,
#             verbose = True):
#         super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
#         self.alpha = alpha
#         self.l1_ratio = l1_ratio
#         self.n_features = None

#     def compute_loss(self, y_true, y_pred) -> float:
#         penalty = elastic_net(self.w, self.l1_ratio)  # weighted sum of l1 norm and l2 norm
#         loss = mse_score(y_true, y_pred) + self.alpha * penalty
#         return loss

#     def gradient_update(self, x, y_true, y_pred) -> None:

#         error = y_true - y_pred
#         m = x.shape[0]

#         gradient_penalty = self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * 2 * self.w
#         gradient_w = -2 * x.T @ error / m + self.alpha * gradient_penalty
#         gradient_b = -2 * error.mean()

#         self.w -= self.learning_rate * gradient_w
#         self.b -= self.learning_rate * gradient_b






# """
# Includes no penalty in normal LinearRegression.

# >>> loss(X, y) = 1/m sum((y - xw - b)^2)
# >>> # Linear Regression partial differentials
# >>> d/dw loss(w, b) = -2/m * x(y - xw - b)
# >>> d/db loss(w, b) =   -2(y - xw - b) 
# """

# """
# Ordinary Least Squares Linear Regression implementaion.

# X is the design matrix, B the coefficient vector and y the obervation
# vector.

# >>> XB = y
# >>> X^T @ XB = X^T @ y
# >>> B = (X^T @ X)^-1 @ X^T y  # !not solvable for singular matrix, IMT
# """ 

# """
# Compute the Elastic Net gradient with L1 and L2 norm as regularisation.

# >>> penalty(w) = l1_ratio * sum(|w|) + (1 - l1_ratio) * sum(w^2)
# >>> loss(X, y) = sum((y - xw - b)^2) + alpha * penalty
# >>> # Elastic net partial differentials
# >>> d/dw penalty(w) = l1_ratio * sign(w) + (1 - l1_ratio) * 2 * w
# >>> d/dw loss(w, b) = -2*x(y - xw - b) + alpha * d/dw penalty(w)
# >>> d/db loss(w, b) =   -2(y - xw - b)
# """


# """
# Compute the LASSO gradient with L1 norm as regularisation.

# >>> loss(X, y) = sum((y - xw - b)^2) + alpha * sum(|w|)
# >>> # LASSO partial differentials
# >>> d/dw loss(w, b) = -2*x(y - xw - b) + alpha * sign(w)
# >>> d/db loss(w, b) =   -2(y - xw - b) 
# """

# """
# Compute the Ridge gradient with L2 norm as regularisation.

# >>> loss(X, y) = sum((y - xw - b)^2) + lamda * sum(w^2)
# >>> # Ridge partial differentials
# >>> d/dw loss(w, b) = -2*x(y - xw - b) + 2 * lamda * w
# >>> d/db loss(w, b) =   -2(y - xw - b) 
# """