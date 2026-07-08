"""
optimizers/constrained_optimizers.py
Constrained Optimization Methods: Projected Gradient Descent and Uzawa's Algorithm.
"""
import numpy as np

def projected_gradient_descent(
    loss_fn, 
    grad_fn, 
    projection_fn, 
    theta_0: np.ndarray, 
    alpha: float, 
    max_iter: int = 1000, 
    tol: float = 1e-6
) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
    # Initialize variables and copy the starting point safely
    theta = theta_0.copy()
    history = {"theta": [theta.copy()], "loss": [loss_fn(theta)]}
    
    for k in range(max_iter):
        # Compute gradient and check termination criterion based on gradient norm
        grad_k = grad_fn(theta)
        if np.linalg.norm(grad_k) < tol:
            break
        
        # 1. Standard Gradient Descent Step (provisional unconstrained update)
        theta_prov = theta - alpha * grad_k
        
        # 2. Geometry Projection Step onto the feasible set
        theta = projection_fn(theta_prov)
        
        # --- RECORD ITERATION DATA IN HISTORY ---
        history["theta"].append(theta.copy())
        history["loss"].append(loss_fn(theta))
        
    return (theta, history)


def uzawa_optimizer_equality(
    Q: np.ndarray, 
    b: np.ndarray, 
    C: np.ndarray, 
    d_vec: np.ndarray, 
    mu_0: np.ndarray, 
    rho: float, 
    max_iter: int = 500,
    tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray | float]]]:
    # Initialize optimization variables
    theta = np.zeros(Q.shape[0])
    mu = mu_0.copy()
    
    # Helper to track the implicit quadratic loss function across iterations:
    # f(theta) = 0.5 * theta^T @ Q @ theta - b^T @ theta
    compute_loss = lambda t: (0.5 * t.T @ Q @ t - b.T @ t).item()
    
    history = {"theta": [], "loss": [compute_loss(theta)], "mu": [mu.copy()]}
    
    for k in range(max_iter):
        # 1. Primal Step: Solve the linear system (Q @ theta = b - C.T @ mu)
        theta = np.linalg.solve(Q, b - C.T @ mu)
        
        # --- RECORD ITERATION DATA IN HISTORY ---
        history["theta"].append(theta.copy())
        history["loss"].append(compute_loss(theta))
        
        # 2. Dual Step: Gradient ascent update on Lagrange multipliers
        # Check convergence based on the linear equality constraint residual (C @ theta - d_vec)
        if np.linalg.norm(C @ theta - d_vec) < tol:  # 💡 Corrigé ici !
            break
            
        mu = mu + rho * (C @ theta - d_vec)
        history["mu"].append(mu.copy())
        
    return (theta, mu, history)