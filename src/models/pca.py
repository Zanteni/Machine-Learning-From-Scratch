"""
transformers/pca.py
Principal Component Analysis (PCA) using our custom SVD implementation.
"""
import numpy as np
from my_ml.linalg.svd import truncated_svd
from my_ml.linalg.svd import svd_from_scratch

class PCA:
    def __init__(self, n_components: int = 2):
        """
        n_components: Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None            # Projection matrix (V_k)
        self.mean = None                  # Empirical mean per feature
        self.std = None                   # Empirical standard deviation per feature
        self.explained_variance_ratio = None  # Percentage of variance explained

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fits the principal components by applying our custom SVD on standardized data.
        Assumes X has shape (n_samples, n_features) with n_samples >= n_features.
        """
        n_samples, n_features = X.shape

        # ── TODO: STEP 1 (Standardization) ────────────────────────────────────
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Handle zero variance to avoid division by zero
        self.std[self.std == 0.0] = 1.0
        X_scaled = (X - self.mean) / self.std

        # ── TODO: STEP 2 (Custom SVD Execution) ───────────────────────────────
        U, Sigma, Vt, sing_values, r = svd_from_scratch(X_scaled)
        
        # ── TODO: STEP 3 (Truncation & Components Extraction) ─────────────────
        k = min(self.n_components, r)
        # Extract rows from Vt and transpose to align axes as columns
        self.components = Vt[:k, :].T
        
        # ── TODO: STEP 4 (Explained Variance Ratio) ───────────────────────────
        eigenvalues = (sing_values ** 2) / (n_samples - 1)
        ratio_n_comp = eigenvalues / np.sum(eigenvalues)
        self.explained_variance_ratio = ratio_n_comp[:k]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """ss
        Projects new data X onto the learned orthogonal principal subspace.
        """
        # ── TODO: STEP 5 (Subspace Projection) ────────────────────────────────
        X_scaled = (X - self.mean) / self.std
        return X_scaled @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the components and returns the projected data in a single pass.
        """
        return self.fit(X).transform(X)