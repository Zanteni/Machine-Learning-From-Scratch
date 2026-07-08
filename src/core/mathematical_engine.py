"""
core/mathematical_engine.py
Moteur analytique universel (Fonctions, Gradients, Hessiennes).
"""
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    
    z = np.clip(z,-500,500)
    return 1/(1+np.exp(-z))
    

def mse_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    
    n = X.shape[0]
    error = X@theta -y
    return (1/(2*n))*(error.T@error).item()
    

def mse_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return (1/n)*X.T@(X@theta-y)
    

def mse_hessian(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return (1/n)*X.T@X
    

def bce_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    n = X.shape[0]
    p = sigmoid(X@theta)
    p = np.clip(p,eps,1-eps)
    
    bce_ls = -(1 / n) * (y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)).item()
    return bce_ls

def bce_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    
    n = X.shape[0]
    return (1/n)*X.T@(sigmoid(X@theta)-y)
    

def bce_hessian(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    p = sigmoid(X@theta)
    W =(p*(1-p)).ravel()
    return (1 / n) * (X.T *W) @ X
   
# ─── Tikhonov — versions régularisées ────────────────────────────────────────

def mse_loss_tikhonov(theta: np.ndarray, X: np.ndarray,
                      y: np.ndarray, lmbda: float) -> float:
    return mse_loss(theta, X, y) + (lmbda / 2) * np.dot(theta.ravel(), theta.ravel())

def mse_gradient_tikhonov(theta: np.ndarray, X: np.ndarray,
                           y: np.ndarray, lmbda: float) -> np.ndarray:
    return mse_gradient(theta, X, y) + lmbda * theta

def mse_hessian_tikhonov(X: np.ndarray, lmbda: float) -> np.ndarray:
    return mse_hessian(X) + lmbda * np.eye(X.shape[1])


def bce_loss_tikhonov(theta: np.ndarray, X: np.ndarray,
                      y: np.ndarray, lmbda: float) -> float:
    return bce_loss(theta, X, y) + (lmbda / 2) * np.dot(theta.ravel(), theta.ravel())

def bce_gradient_tikhonov(theta: np.ndarray, X: np.ndarray,
                           y: np.ndarray, lmbda: float) -> np.ndarray:
    return bce_gradient(theta, X, y) + lmbda * theta

def bce_hessian_tikhonov(theta: np.ndarray, X: np.ndarray,
                          lmbda: float) -> np.ndarray:
    return bce_hessian(theta, X) + lmbda * np.eye(X.shape[1])
