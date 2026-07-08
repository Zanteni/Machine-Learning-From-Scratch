"""
optimizers/newton.py
Newton-Raphson optimization method.
"""
import numpy as np
from my_ml.line_search.line_search import armijo_backtracking

class NewtonOptimizer:
    def __init__(self, loss_fn, grad_fn, hess_fn):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn

    def optimize(self, theta_0: np.ndarray, use_line_search: bool = False, tol: float = 1e-6, max_iter: int = 100) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
        # Initialize parameters and copy the starting point safely
        theta = theta_0.copy()
        history = {"theta": [theta.copy()], "loss": [self.loss_fn(theta)]}
        
        for k in range(max_iter):
            # Compute gradient and check termination criterion
            grad_k = self.grad_fn(theta)
            if np.linalg.norm(grad_k) < tol:
                break
                
            # Compute Hessian matrix at the current point
            hess_k = self.hess_fn(theta)
            
            # 1. Compute Newton direction d_k by solving H_k * d_k = -g_k (more stable than inverse)
            d_k = np.linalg.solve(hess_k, -grad_k)
            
            # 2. Determine step size alpha_k (using Armijo line search or pure Newton step of 1.0)
            if use_line_search:
                alpha_k = armijo_backtracking(self.loss_fn, theta, d_k, grad_k)
            else:
                alpha_k = 1.0
                
            # 3. Update position vector theta
            theta += alpha_k * d_k
            
            
            history["theta"].append(theta.copy())
            history["loss"].append(self.loss_fn(theta))
            
        return (theta, history)