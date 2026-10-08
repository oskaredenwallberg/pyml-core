import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.base import Estimator, Linear
from pyml.base import Optimizer
from pyml.optimization import GradientDescent
from pyml.config.constants import EPS


class LogisticRegression(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            optimizer: Optimizer,
        ):
        self.lamda = lamda
        self.optimizer = optimizer
        self.params: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        params = np.zeros(F+1)
        params = self.optimizer.run(self, x, y, params)
        self.params = params

    def loss(self, x: NDArray, y: NDArray, theta: NDArray) -> float:
        N = y.size
        p = sigmoid(x @ theta)
        loss = -1/N * (y.T @ np.log(p + EPS) + (1-y).T @ np.log(1-p + EPS))
        loss += self.lamda * np.sum(theta[1:] ** 2)
        return loss

    def gradient(self, x: NDArray, y: NDArray, theta: NDArray) -> NDArray:
        N = y.size
        p = sigmoid(x @ theta)
        grad = 1/N * x.T @ ( p - y )
        grad[1:] += 2 * self.lamda * theta[1:]
        return grad

    def hessian(self, x: NDArray, y: NDArray, theta: NDArray) -> NDArray:
        ... # for newton method

    def predict(self, x):
        return sigmoid(self.linear(x))

    def probability(self, x):
        return sigmoid(self.linear(x)).round(0)

    def prb(self, x):
        return self.probability(x)


def sigmoid(z: ArrayLike) -> NDArray:
    return 1 / (1 + np.exp(-z))


class LogisticRegressionGD(LogisticRegression):
    def __init__(
            self,
            lamda: float,
            batch_size: int | None = None,
            iterations: int = 100,
            tolerance: float = 1e-4,
            learning_rate: float = 1e-3,
        ):
        optimizer = GradientDescent(
            batch_size, 
            iterations, 
            tolerance, 
            learning_rate
        )
        super().__init__(lamda=lamda, optimizer=optimizer)






"""
## Scalar and summation form

likelihood l
l = prod( p^y*(1-p)^(1-y) )^{1/n}  # Bernoulli likelihood for all examples
nll = -log l
nll = -1/n sum( y*log(p) + (1-y)log(1-p) )

p = sigmoid(z) = 1 / (1 + e^{-z})
z = Xw + b

dz/dw = d/dw Xw + b = X
dz/db = d/db Xw + b = 1

d/dw nll = d/dp*dp/dz*dz/dw nll
d/dp nll = -1/n sum( y/p - (1-y)/(1-p) )
dp/dz 
 = e^{-z} / (1 + e^{-z})^2 
 = (1 + e^{-z}) / (1 + e^{-z})^2 - 1/(1 + e^{-z})^2
 = 1 / (1 + e^{-z}) - 1/(1 + e^{-z})^2
 = p - p^2
 = p(1 - p)
d/dz nll = d/dp*dp/dz nll
 = -1/n sum( y/p - (1-y)/(1-p) ) * p(1 - p)
 = -1/n sum( y(1 - p) - (1-y)p )
 = -1/n sum( y - yp - p + yp )
 = -1/n sum( y - p )
 = -1/n sum( y - 1 / (1 + e^{-z}) )
d/dw nll = d/dz*dz/dw nll
 = -1/n X^T (y - 1 / (1 + e^{-Xw-b}))
d/db nll = d/dz*dz/db nll
 = -1/n 1^T (y - 1 / (1 + e^{-Xw-b}))

 
## Vectorized form

p = 1 / (1 + e^{-z})
J_{p,z} 
 = diag( e^{-z} / (1 + e^{-z})^2 )
 = diag( p - p^2 )

J_{z,w} = X
J_{z,b} = 1  in  R^{n}

nll 
 = -1/n ( y^T log(p) + (1-y)^T log(1-p) )  in  R
nabla_p nll
 = -1/n ( y/p - (1-y)/(1-p) )
nabla_z nll 
 = J_{p,z}^T nabla_p nll
 = diag(p - p^2)^T -1/n ( y/p - (1-y)/(1-p) )
 = -1/n ( diag(p - p^2)^T y/p - diag(p - p^2)^T (1-y)/(1-p) )
 = -1/n ( diag(1 - p)^T y - diag(p)^T (1-y) )
 = -1/n ( y - y*p - p + y*p )
 = -1/n ( y - p )
 = -1/n ( y - p )  in  R^{n}
nabla_w nll 
 = J_{z,w}^T nabla_z nll
 = -1/n X^T ( y - p )
 = 1/n X^T (p - y)
nabla_b nll
 = J_{z,b}^T nabla_z nll
 = -1/n 1^T ( y - p )
 = 1/n 1^T (p - y)

With l2 penalty:
nabla_w nll = 1/n X^T (p - y) + 2 lamda w
nabla_b nll = 1/n 1^T (p - y)
"""

# multi class classification extension of logistic regression
class SoftmaxRegressionGD(Estimator, Linear):
    pass # TODO
