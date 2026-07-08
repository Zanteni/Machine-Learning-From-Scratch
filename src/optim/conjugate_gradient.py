"""
optimizers/conjugate_gradient.py
Linear Conjugate Gradient (quadratic) and Non-Linear (Fletcher-Reeves).
"""
import numpy as np
from my_ml.line_search.line_search import armijo_backtracking, exact_line_search_quadratic

class ConjugateGradientOptimizer:
    def __init__(self, loss_fn, grad_fn):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn

    def optimize_quadratic(self, theta_0: np.ndarray, Q: np.ndarray, b: np.ndarray, tol: float = 1e-6, max_iter: int = None) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
        # Solve the symmetric positive-definite system Q @ theta = b (Krylov subspace).
        history = {"theta": [theta_0.copy()], "loss": [self.loss_fn(theta_0)]}

        if max_iter is None:
            max_iter = Q.shape[0]
            
        theta = theta_0.copy()
        r = b - Q @ theta
        d = r.copy()

        for k in range(max_iter):
            # Check convergence based on the residual vector norm
            if np.linalg.norm(r) < tol:
                break
                
            # Compute exact step size alpha_k for quadratic form
            alpha_k = exact_line_search_quadratic(Q, -r, d)
            theta = theta + alpha_k * d
            
            # --- RECORD ITERATION DATA IN HISTORY ---
            history["theta"].append(theta.copy())  # Added .copy() here to protect array reference
            history["loss"].append(self.loss_fn(theta))
            
            # Update residual vector and compute beta_k parameter (Fletcher-Reeves style)
            r_new = r - alpha_k * Q @ d
            num_beta = (r_new.T @ r_new).item()
            den_beta = (r.T @ r).item()
            beta_k = num_beta / den_beta
            
            # Update conjugate search direction for the next iteration
            d = r_new + beta_k * d
            r = r_new
            
        return (theta, history)

    def optimize_nonlinear_fr(self, theta_0: np.ndarray, restart_interval: int = 10, tol: float = 1e-6, max_iter: int = 500) -> tuple[np.ndarray, dict[str, list[np.ndarray | float]]]:
            # Non-linear Fletcher-Reeves algorithm with Armijo backtracking line search.
            theta = theta_0.copy()
            history = {"theta": [theta.copy()], "loss": [self.loss_fn(theta)]}
            
            grad_k = self.grad_fn(theta)
            d_k = -grad_k.copy()
            
            for k in range(max_iter):
                # Condition d'arrêt sur la norme du gradient
                if np.linalg.norm(grad_k) < tol:
                    break
                    
                # 1. Recherche linéaire d'Armijo
                alpha_k = armijo_backtracking(self.loss_fn, theta, d_k, grad_k)
                
                # 2. Mise à jour des paramètres
                theta = theta + alpha_k * d_k
                
                # CORRIGÉ : Ajout du .copy() pour figer les valeurs dans l'historique
                history["theta"].append(theta.copy())
                history["loss"].append(self.loss_fn(theta))
                
                # 3. Calcul du gradient à l'étape k+1
                grad_k_plus = self.grad_fn(theta)
                
                # CORRIGÉ : Implémentation du mécanisme de Restart (Fletcher-Reeves)
                if (k + 1) % restart_interval == 0:
                    beta_k = 0.0  # Réinitialisation (pas de mémoire des étapes précédentes)
                else:
                    beta_num = (grad_k_plus.T @ grad_k_plus).item()
                    beta_den = (grad_k.T @ grad_k).item()
                    # Sécurité mathématique contre la division par zéro
                    beta_k = beta_num / beta_den if beta_den > 1e-12 else 0.0
                
                # 4. Calcul de la nouvelle direction conjuguée
                d_k = -grad_k_plus + beta_k * d_k
                
                # Passation des variables pour l'itération suivante
                grad_k = grad_k_plus
                
            return (theta, history)