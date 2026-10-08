import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator


class BFGS(Optimizer):
    def __init__(
            self,
            iterations_max: int,
            tolerance: float,
        ):
        self.iterations_max = iterations_max
        self.tolerance = tolerance
        self.strong_wolfe = StrongWolfe(c1=1e-4, c2=0.9, iterations_max=20)

    def run(self, estimator, x, y, params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        F = params.size
        If = np.eye(F)

        gk = estimator.gradient(x, y, params)
        Bk = np.eye(F)  # inverse hessian approximation
        params_k = params

        for i in range(0, self.iterations_max):
            if self.converged(gk) == True:
                break

            if i > 0:
                grad = estimator.gradient(x, y, params)
                sk = params - params_k
                yk = grad - gk
                pk = 1.0 / (yk @ sk)
                gk = grad
                params_k = params

                Ak = If - pk * np.outer(yk, sk)
                Ck = pk * np.outer(sk, sk)
                Bk = Ak.T @ Bk @ Ak + Ck

            dk = - Bk @ gk  # update direction
            alpha = self.strong_wolfe.line_search(estimator, x, y, params_k, dk)
            params = params + alpha * dk
            losses[i] = estimator.loss(x, y, params)

        self.losses = losses
        return params

    def converged(self, grad: NDArray) -> bool:
        return np.linalg.norm(grad, ord=2) <= self.tolerance


class StrongWolfe:
    def __init__(
            self,
            c1: float = 1e-4,
            c2: float = 0.9,
            iterations_max = 20,
        ):
        assert 0 < c1 < c2 < 1, (c1, c2)
        self.c1 = c1
        self.c2 = c2
        self.iterations_max = iterations_max

    def line_search(
            self,
            estimator: Estimator,
            x: NDArray,
            y: NDArray,
            params: NDArray,
            dk: NDArray,  # direction
        ) -> float:
        alpha_lo = 0
        alpha_hi = np.inf
        alpha = 1

        phi_0, phi_d0 = self.phi(estimator, x, y, params, 0, dk)
        assert phi_d0 < 0, phi_d0
        
        for i in range(self.iterations_max):  # bracket
            phi_a, phi_da = self.phi(estimator, x, y, params, alpha, dk)

            if self.armijo(phi_0, phi_d0, phi_a, alpha) == False:
                alpha_hi = alpha
                break

            if self.swolfe(phi_d0, phi_da) == True:
                return alpha

            if phi_da <= 0:
                alpha_lo = alpha
                alpha *= 2
            else:
                alpha_hi = alpha
                break

        assert np.isfinite(alpha_hi), alpha_hi

        for i in range(self.iterations_max):  # zoom
            alpha = 0.5 * (alpha_hi + alpha_lo)
            phi_a, phi_da = self.phi(estimator, x, y, params, alpha, dk)

            if self.armijo(phi_0, phi_d0, phi_a, alpha) == False:
                alpha_hi = alpha
                continue

            if self.swolfe(phi_d0, phi_da) == True:
                return alpha
            
            if phi_da <= 0:
                alpha_lo = alpha
            else:
                alpha_hi = alpha

        raise RuntimeError

    def phi(self, estimator, x, y, params, alpha, dk):
        params_x = params + alpha * dk
        phi_x = estimator.loss(x, y, params_x)
        phi_dx = estimator.gradient(x, y, params_x) @ dk
        return phi_x, phi_dx

    def armijo(self, phi_0, phi_d0, phi_a, alpha) -> bool:
        # phi(alpha) - phi(0) <= c1 alpha phi'(0)
        return phi_a - phi_0 <= self.c1 * alpha * phi_d0
        
    def swolfe(self, phi_d0, phi_da) -> bool:
        # |phi'(alpha)| <= c2|phi'(0)|
        return np.abs(phi_da) <= self.c2 * np.abs(phi_d0)


# params_a = params + alpha * dk
# phi_a = estimator.loss(x, y, params_a)
# phi_da = estimator.gradient(x, y, params_a) @ dk
