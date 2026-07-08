"""
linalg/qr.py
QR Factorization and associated linear system resolution algorithms.
"""
import numpy as np
from my_ml.linalg.gram_schmidt import modified_gram_schmidt

def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the QR factorization of A using Modified Gram-Schmidt."""
    Q = modified_gram_schmidt(A)
    R = Q.T @ A
    return (Q, R)


def back_substitution(R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Solves the upper triangular system R @ theta = c using back-substitution.
    Safely checks for matrix singularity and leverages NumPy vectorization.
    """
    R_diag = np.diag(R)
    tol = 1e-12
    
    # Robust Singularity Check
    if np.any(np.abs(R_diag) < tol):
        raise ValueError("Matrix R is singular or nearly singular (zero diagonal detected).")
    
    n, _ = R.shape
    theta = np.zeros((n, 1))
    
    # Base case: compute the very last parameter theta_n
    theta[n-1, 0] = c[n-1, 0] / R_diag[n-1]
    
    # Backward substitution loop
    for i in range(n-2, -1, -1):
        theta_founded = theta[i+1:]  # Shape: (k, 1)
        R_needed = R[i, i+1:].reshape(1, -1)  # Shape: (1, k)
        
        # Row (1, k) @ Column (k, 1) -> Matrix (1, 1) -> .item() extracts the scalar
        c_needed = c[i, 0] - (R_needed @ theta_founded).item()
        theta[i, 0] = c_needed / R_diag[i]

    return theta