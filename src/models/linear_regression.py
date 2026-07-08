"""
models/linear_regression.py
Linear Regression with Tikhonov regularization and multi-optimizer dispatching.
"""
import numpy as np
from my_ml.core.mathematical_engine import mse_loss_tikhonov, mse_gradient_tikhonov, mse_hessian_tikhonov
from my_ml.linalg.qr import qr_decomposition, back_substitution
from my_ml.linalg.pseudo_inverse import pseudo_inverse_moore_penrose
from my_ml.optimizers.gradient_descent import GradientDescentOptimizer
from my_ml.optimizers.conjugate_gradient import ConjugateGradientOptimizer
from my_ml.optimizers.newton import NewtonOptimizer

class LinearRegression:
    def __init__(self,
                 method: str = "normal",
                 alpha: float = 0.01,
                 lambd: float = 0.0,
                 tol: float = 1e-6,
                 max_iter: int = 1000,
                 add_bias: bool = True,
                 line_search_method: str = None,
                 batch_size: int = 24,
                 epoch: int = 50):
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.lambd = lambd
        self.add_bias = add_bias  # Fixed: corrected spelling from 'bais'
        self.line_search_method = line_search_method
        self.batch_size = batch_size
        self.epoch = epoch
        self.theta = None
        self.history = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_train = self._add_bias(X) if self.add_bias else X
        y_train = y.reshape(-1, 1) if y.ndim == 1 else y  # Safety: enforce (n, 1) column format
        
        # Core functional closures linked to the mathematical engine
        loss_fn = lambda t: mse_loss_tikhonov(t, X_train, y_train, self.lambd)
        grad_fn = lambda t: mse_gradient_tikhonov(t, X_train, y_train, self.lambd)
        hess_fn = lambda t: mse_hessian_tikhonov(X_train, self.lambd)
        
        n, d = X_train.shape
        theta_0 = np.random.randn(d, 1) * 0.01
        rank = np.linalg.matrix_rank(X_train)
        
        # 1. Closed-Form Analytical Solution via Normal Equations
        if self.method == "normal":
            if rank == d:
                # QR resolution of the regularized system: (X^T X + lambda I) theta = X^T y
                A = X_train.T @ X_train + self.lambd * np.eye(d)
                b = X_train.T @ y_train
                Q, R = qr_decomposition(A)
                c = Q.T @ b
                self.theta = back_substitution(R, c)
            else:
                # Underdetermined or singular case: fallback to Moore-Penrose Pseudo-Inverse
                self.theta = pseudo_inverse_moore_penrose(X_train) @ y_train
            
            self.history = {"theta": [self.theta.copy()], "loss": [loss_fn(self.theta)]}
            
        # 2. Standard Fixed-Step Gradient Descent
        elif self.method == "gd_fixed":
            opt = GradientDescentOptimizer(loss_fn, grad_fn)
            # Fixed: named arguments prevent accidental tol/max_iter inversion
            self.theta, self.history = opt.optimize_fixed(theta_0, self.alpha, tol=self.tol, max_iter=self.max_iter)
            
        # 3. Backtracking Variable-Step Gradient Descent (Line Search)
        elif self.method == "gd_variable":
            line_search_method = self.line_search_method if self.line_search_method else "armijo"
            opt = GradientDescentOptimizer(loss_fn, grad_fn)
            self.theta, self.history = opt.optimize_variable(theta_0, line_search_method, max_iter=self.max_iter, tol=self.tol)  
            
        # 4. Steepest Descent (Exact Optimal Step Size for Quadratic Forms)
        elif self.method == "gd_optimal":
            opt = GradientDescentOptimizer(loss_fn, grad_fn)
            # Fixed: added Tikhonov penalty to the curvature matrix Q
            Q = X_train.T @ X_train + self.lambd * np.eye(d)
            self.theta, self.history = opt.optimize_optimal(theta_0, Q, tol=self.tol, max_iter=self.max_iter)
            
        # 5. Linear Conjugate Gradient Method
        elif self.method == "gc":
            opt = ConjugateGradientOptimizer(loss_fn, grad_fn)
            # Fixed: added Tikhonov penalty to the curvature matrix Q and right-hand side b
            Q = X_train.T @ X_train + self.lambd * np.eye(d)
            b = X_train.T @ y_train
            self.theta, self.history = opt.optimize_quadratic(theta_0, Q, b, tol=self.tol, max_iter=self.max_iter)
            
        # 6. Pure Second-Order Newton's Method
        elif self.method == "newton":
            opt = NewtonOptimizer(loss_fn, grad_fn, hess_fn)
            # Line search skipped: MSE is purely quadratic and strictly convex
            self.theta, self.history = opt.optimize(theta_0, use_line_search=False, tol=self.tol, max_iter=self.max_iter)
            
        # 7. Mini-Batch Stochastic Gradient Descent (SGD)
        elif self.method == "sgd":
            grad_fn_sgd = lambda t, xb, yb: mse_gradient_tikhonov(t, xb, yb, self.lambd)
            opt = GradientDescentOptimizer(loss_fn, grad_fn)
            # Fixed: correctly assigned learning rate to alpha (instead of max_iter)
            self.theta, self.history = opt.optimize_stochastic(
                theta_0, X_train, y_train, alpha=self.alpha, 
                batch_size=self.batch_size, epochs=self.epoch, 
                custom_grad_fn=grad_fn_sgd
            )
        else:
            raise ValueError("Unknown method. Choose from: ['normal', 'gd_fixed', 'gd_variable', 'gd_optimal', 'sgd', 'gc', 'newton']")
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_proc = self._add_bias(X) if self.add_bias else X
        return X_proc @ self.theta