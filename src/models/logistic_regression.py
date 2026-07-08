"""
models/logistic_regression.py
Logistic Regression (Binary Classification) with Tikhonov regularization.
"""
import numpy as np
from my_ml.core.mathematical_engine import (
    sigmoid, bce_loss_tikhonov, bce_gradient_tikhonov, bce_hessian_tikhonov
)
from my_ml.optimizers.gradient_descent import GradientDescentOptimizer
from my_ml.optimizers.newton import NewtonOptimizer

class LogisticRegression:
    def __init__(self, 
                 method: str = "gd_fixed", 
                 lmbda: float = 0.0,
                 tol: float = 1e-6,
                 max_iter: int = 1000,
                 alpha: float = 0.01,
                 add_bias: bool = True):
        self.method = method
        self.lmbda = lmbda
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.add_bias = add_bias
        self.theta = None
        self.history = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        # ── 1. Data Processing (Scoped locally for memory optimization) ───────
        X_train = self._add_bias(X) if self.add_bias else X
        y_train = y.reshape(-1, 1) if y.ndim == 1 else y
        
        n, d = X_train.shape
        # Scaled initialization crucial to avoid early sigmoid saturation
        theta_0 = np.random.randn(d, 1) * 0.01  

        # ── 2. Functional Closures (BCE Loss + Tikhonov Regularization) ───────
        loss_fn = lambda t: bce_loss_tikhonov(t, X_train, y_train, self.lmbda)
        grad_fn = lambda t: bce_gradient_tikhonov(t, X_train, y_train, self.lmbda)
        hess_fn = lambda t: bce_hessian_tikhonov(t, X_train, self.lmbda)
        
        # ── 3. Optimizer Dispatcher ───────────────────────────────────────────
        if self.method == "gd_fixed":
            opt = GradientDescentOptimizer(loss_fn, grad_fn)
            self.theta, self.history = opt.optimize_fixed(
                theta_0, self.alpha, tol=self.tol, max_iter=self.max_iter
            )

        elif self.method == "newton":
            opt = NewtonOptimizer(loss_fn, grad_fn, hess_fn)
            # Pure Newton (use_line_search=False) works due to local strict convexity
            self.theta, self.history = opt.optimize(
                theta_0, use_line_search=False, tol=self.tol, max_iter=self.max_iter
            )

        else:
            raise ValueError(f"Unknown method='{self.method}'. Choose from: gd_fixed | newton")

        return self

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Returns empirical probability scores using the sigmoid function."""
        X_proc = self._add_bias(X) if self.add_bias else X
        return sigmoid(X_proc @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Maps probabilities to hard binary classifications {0, 1}."""
        return (self.predict_prob(X) >= threshold).astype(int)