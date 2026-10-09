import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator


class BFGS(Optimizer):  # Broyden Fletcher Goldfarb Shanno
    def __init__(
            self,
            iterations_max: int,
            tolerance: float,
        ):
        self.iterations_max = iterations_max
        self.tolerance = tolerance
        self.line_search = StrongWolfe(c1=1e-4, c2=0.9, iterations_max=20)

    def run(self, estimator, x, y, params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        F = params.size
        If = np.eye(F)

        tk = params
        gk = estimator.gradient(x, y, params)  # gk
        Bk = np.eye(F)  # inverse hessian approximation

        for i in range(0, self.iterations_max):
            if self.converged(gk) == True:
                break

            pk = - Bk @ gk  # update direction
            alpha = self.line_search.run(estimator, x, y, tk, pk)
            params = params + alpha * pk
            grad = estimator.gradient(x, y, params)

            Bk = self.inverse_hessian_update(If, Bk, tk, params, gk, grad)
            gk = grad
            tk = params

            losses[i] = estimator.loss(x, y, params)

        self.losses = losses
        return params

    def inverse_hessian_update(
            self, 
            If: NDArray, Bk: NDArray,
            tk: NDArray, tk1: NDArray, 
            gk: NDArray, gk1: NDArray,
        ) -> NDArray:
        sk = tk1 - tk
        yk = gk1 - gk
        ys = yk @ sk
        assert ys > 0, ys
        rhok = 1.0 / ys
        Ak = If - rhok * np.outer(yk, sk)
        Ck = rhok * np.outer(sk, sk)
        Bk = Ak.T @ Bk @ Ak + Ck
        return Bk
        
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

    def run(
            self,
            estimator: Estimator,
            x: NDArray,
            y: NDArray,
            params: NDArray,
            pk: NDArray,  # direction
        ) -> float:

        phi  = lambda alpha: estimator.loss(x, y, params + alpha * pk)
        dphi = lambda alpha: estimator.gradient(x, y, params + alpha * pk) @ pk

        alpha_lo, alpha_hi = self.bracket(phi, dphi)
        if alpha_hi == alpha_lo:
            return alpha_hi
        
        alpha = self.zoom(phi, dphi, alpha_lo, alpha_hi)
        return alpha
    
    def bracket(self, phi, dphi):
        alpha_lo = 0
        alpha_hi = np.inf
        alpha = 1
        phi_0  = phi(0)
        dphi_0 = dphi(0)
        
        for _ in range(self.iterations_max):  # bracket
            phi_alpha  = phi(alpha)
            dphi_alpha = dphi(alpha)

            if not self.armijo(phi_0, dphi_0, phi_alpha, alpha):
                alpha_hi = alpha
                break

            if not self.strong_wolfe(dphi_0, dphi_alpha):
                if dphi_alpha > 0:
                    alpha_hi = alpha
                    break
                else:
                    alpha_lo = alpha
                    alpha *= 2
                    continue

            return alpha, alpha
        
        return alpha_lo, alpha_hi

    def zoom(self, phi, dphi, alpha_lo, alpha_hi):
        assert np.isfinite(alpha_hi), alpha_hi
        phi_0  = phi(0)
        dphi_0 = dphi(0)

        for _ in range(self.iterations_max):  # zoom
            alpha = 0.5 * (alpha_hi + alpha_lo)
            phi_alpha  = phi(alpha)
            dphi_alpha = dphi(alpha)

            if not self.armijo(phi_0, dphi_0, phi_alpha, alpha):
                alpha_hi = alpha
                continue

            if not self.strong_wolfe(dphi_0, dphi_alpha):
                if dphi_alpha > 0:
                    alpha_hi = alpha
                else:
                    alpha_lo = alpha
                continue
            
            return alpha

        raise RuntimeError

    def armijo(self, phi_0, dphi_0, phi_alpha, alpha) -> bool:  
        return phi_alpha - phi_0 <= self.c1 * alpha * dphi_0
        # phi(alpha) - phi(0) <= c1 alpha phi'(0)

    def strong_wolfe(self, dphi_0, dphi_alpha) -> bool:  
        return np.abs(dphi_alpha) <= self.c2 * np.abs(dphi_0)
        # |phi'(alpha)| <= c2|phi'(0)|
