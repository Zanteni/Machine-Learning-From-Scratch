"""
linalg/gram_schmidt.py
Vector family orthogonalization: Classical vs Modified Gram-Schmidt algorithms
with rank-deficiency failure recovery (assuming m >= n).
"""
import numpy as np

def classical_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    # Classical Gram-Schmidt (CGS) with canonical basis fallback recovery
    m, n = A.shape
    Q = np.zeros((m, n))
    basis_m = np.eye(m)  # Standard canonical basis to catch rank deficiency
    
    for j in range(n):
        v_j = A[:, j].copy()
        
        if j > 0:
            # Multi-projection: simultaneous projection onto all previous vectors
            U_computed = Q[:, :j]
            v_j -= U_computed @ (U_computed.T @ v_j)
            
        v_j_norm = np.linalg.norm(v_j)
        
        # Scenario A: Vector is valid and linearly independent
        if v_j_norm > tol:
            q_j = v_j / v_j_norm
        else:
            # Scenario B: FAIL SAFE (Rank deficiency detected in CGS)
            for k in range(m):
                e_k = basis_m[:, k].copy()
                if j > 0:
                    U_computed = Q[:, :j]
                    e_k -= U_computed @ (U_computed.T @ e_k)
                
                e_k_norm = np.linalg.norm(e_k)
                if e_k_norm > tol:
                    q_j = e_k / e_k_norm
                    break  # Found a valid direction, exit the fallback loop
                    
        Q[:, j] = q_j
        
    return Q


def modified_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    # Modified Gram-Schmidt (MGS) with canonical basis fallback recovery
    m, n = A.shape
    V = A.astype(float).copy()
    Q = np.zeros((m, n))
    basis_m = np.eye(m)  # Standard canonical basis to catch rank deficiency
    
    for i in range(n):
        v_i = V[:, i].copy()
        v_i_norm = np.linalg.norm(v_i)
        
        # Scenario A: Vector is valid and linearly independent
        if v_i_norm > tol:
            q_i = v_i / v_i_norm
        else:
            # Scenario B: FAIL SAFE (Rank deficiency detected in MGS)
            for j in range(m):
                e_j = basis_m[:, j].copy()
                if i > 0:
                    U_computed = Q[:, :i]
                    e_j -= U_computed @ (U_computed.T @ e_j)
                
                e_j_norm = np.linalg.norm(e_j)
                if e_j_norm > tol:
                    q_i = e_j / e_j_norm
                    break  # Found a valid direction, exit the fallback loop
                    
        Q[:, i] = q_i
        
        # Immediate elimination step for all remaining vectors
        for j in range(i + 1, n):
            v_j = V[:, j]
            V[:, j] = v_j - q_i * (q_i.T @ v_j)
            
    return Q