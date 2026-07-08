"""
linalg/pseudo_inverse.py
Moore-Penrose Pseudo-inverse implementation using SVD.
"""
import numpy as np
from my_ml.linalg.svd import svd_from_scratch

def pseudo_inverse_moore_penrose(X: np.ndarray) -> np.ndarray:
    """
    Computes the Moore-Penrose pseudo-inverse (X^+) using SVD.
    Handles rectangular and rank-deficient matrices safely.
    
    Parameters:
    -----------
    X : np.ndarray
        Input matrix of shape (m, n)
    
    Returns:
    --------
    np.ndarray
        Pseudo-inverse X^+ of shape (n, m)
    
    Theory:
    -------
    X^+ = V @ Sigma^+ @ U^T
    where Sigma^+ has 1/sigma_i on diagonal for non-zero singular values
    """
    U, Sigma, Vt, sing_values, r = svd_from_scratch(X)
    
    # Sigma_plus shape must be the transpose of Sigma: (n x m)
    Sigma_plus = np.zeros_like(Sigma.T)
    
    # Invert only the non-zero singular values up to rank r
    Sigma_plus[:r, :r] = np.diag(1.0 / sing_values)
    
    # Reconstruct using the property: X^+ = V @ Sigma^+ @ U^T
    return Vt.T @ Sigma_plus @ U.T
