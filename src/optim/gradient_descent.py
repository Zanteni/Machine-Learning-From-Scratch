"""
optimizers/gradient_descent.py
Gradient descent algorithms: fixed step, variable step (Armijo/Goldstein), and SGD.
"""
import numpy as np
from my_ml.line_search.line_search import armijo_backtracking, goldstein_line_search

class GradientDescentOptimizer:
    def __init__(self, loss_fn, grad_fn):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn

    def optimize_fixed(self, theta_0: np.ndarray, alpha: float, tol: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
        # Standard Gradient Descent with a fixed step size
        history = {"theta":[theta_0.copy()],"loss":[self.loss_fn(theta_0)]}
        theta = theta_0.copy()
        for k in range(max_iter):
            grad_k = self.grad_fn(theta)
            if np.linalg.norm(grad_k) < tol:
                break
            theta -= alpha * grad_k
            history["theta"].append(theta.copy())  # Ajout de .copy() pour fixer l'historique
            history["loss"].append(self.loss_fn(theta))
        return (theta,history)

    def optimize_variable(self, theta_0: np.ndarray, method: str = "armijo", max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
        # Select the appropriate line search method
        if method == "armijo":
            line_search_fn = armijo_backtracking
        elif method == "goldstein":
            line_search_fn = goldstein_line_search
        else:
            raise ValueError(f"method {method} is not recognized! Choose 'armijo' or 'goldstein'")
        history = {"theta":[theta_0.copy()],"loss":[self.loss_fn(theta_0)]}
        theta = theta_0.copy()  # Ajout de .copy() pour préserver theta_0
        for k in range(max_iter):
            grad_k = self.grad_fn(theta)
            if np.linalg.norm(grad_k) < tol:
                break
            alpha_k = line_search_fn(self.loss_fn, theta, -grad_k, grad_k)
            theta -= alpha_k * grad_k
            history["theta"].append(theta.copy())  # Ajout de .copy() pour fixer l'historique
            history["loss"].append(self.loss_fn(theta))
        return (theta,history)

    def optimize_stochastic(self, theta_0: np.ndarray, X: np.ndarray, y: np.ndarray, 
                           alpha: float, batch_size: int = 1, epochs: int = 100,
                           custom_grad_fn = None) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
        """
        Stochastic Gradient Descent (Mini-batching with shuffling & Robbins-Monro decay).
        
        Parameters:
        -----------
        theta_0 : np.ndarray
            Initial parameters (d, 1)
        X : np.ndarray
            Training features (m, d)
        y : np.ndarray
            Training targets (m, 1)
        alpha : float
            Initial learning rate
        batch_size : int
            Batch size for mini-batches
        epochs : int
            Number of passes through the data
        custom_grad_fn : callable
            Custom gradient function with signature (theta, X_batch, y_batch).
            If None, uses self.grad_fn(theta) evaluated on full X, y
        
        Returns:
        --------
        theta : np.ndarray
            Optimized parameters
        history : dict
            Dictionary with "theta" and "loss" lists
        """
        theta = theta_0.copy()
        m = X.shape[0]
        
        # Initialize history: evaluate loss on FULL dataset initially
        history = {"theta": [theta.copy()], "loss": [self.loss_fn(theta)]}
        
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Robbins-Monro decay: alpha_k decreases over epochs
            decay = 0.1
            alpha_k = alpha / (1 + decay * epoch)
            
            # Iterate over mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                # Use custom gradient if provided, otherwise full-batch gradient
                if custom_grad_fn is not None:
                    grad_k = custom_grad_fn(theta, X_batch, y_batch)
                else:
                    # Fallback: use full-batch gradient (less efficient but safe)
                    grad_k = self.grad_fn(theta)
                
                theta -= alpha_k * grad_k
            
            # Record history at each EPOCH on FULL dataset
            history["theta"].append(theta.copy())
            history["loss"].append(self.loss_fn(theta))
        
        return (theta, history)
    
    def optimize_optimal(self, theta_0: np.ndarray, Q: np.ndarray, tol: float = 1e-6, 
                        max_iter: int = 1000) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
        """
        Steepest Descent with exact optimal step size for quadratic forms.
        
        For a quadratic loss f(θ) = 0.5 * θ^T @ Q @ θ - b^T @ θ,
        the optimal step is: α_k = (g_k^T @ g_k) / (g_k^T @ Q @ g_k)
        
        Parameters:
        -----------
        theta_0 : np.ndarray
            Initial parameters
        Q : np.ndarray
            Hessian matrix (curvature)
        tol : float
            Tolerance for gradient norm convergence
        max_iter : int
            Maximum iterations
        
        Returns:
        --------
        theta : np.ndarray
            Optimized parameters
        history : dict
            Convergence history
        """
        history = {"theta": [theta_0.copy()], "loss": [self.loss_fn(theta_0)]}
        theta = theta_0.copy()
        
        for k in range(max_iter):
            grad_k = self.grad_fn(theta)
            
            # Convergence check
            if np.linalg.norm(grad_k) < tol:
                break
            
            # Compute optimal step size
            numerator = (grad_k.T @ grad_k).item()
            denominator = (grad_k.T @ Q @ grad_k).item()
            
            if abs(denominator) < 1e-12:
                print(f"Warning: Q is nearly singular at iteration {k}")
                break
            
            alpha_k = numerator / denominator
            
            # Update theta
            theta -= alpha_k * grad_k
            
            history["theta"].append(theta.copy())
            history["loss"].append(self.loss_fn(theta))
        
        return (theta, history)