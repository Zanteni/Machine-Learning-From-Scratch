"""
src/preprocessing.py
Data preprocessing utilities: normalization, standardization, train/test split, etc.
"""
import numpy as np

class StandardScaler:
    """
    Standardizes features by removing mean and scaling to unit variance.
    
    X_scaled = (X - mean) / std
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scales features to a fixed range [0, 1].
    
    X_scaled = (X - X_min) / (X_max - X_min)
    """
    
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute min and max from training data."""
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply min-max scaling."""
        if self.min is None or self.max is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        range_vals = self.max - self.min
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero
        
        return (X - self.min) / range_vals
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

def handle_missing_values(X: np.ndarray, y: np.ndarray = None, strategy: str = "mean") -> tuple:
    """
    Handles missing values (NaN) in the dataset.
    """
    if strategy == "mean":
        col_means = np.nanmean(X, axis=0)
        mask = np.isnan(X)
        X_clean = X.copy()
        X_clean = np.where(mask, col_means, X_clean)
    
    elif strategy == "median":
        col_medians = np.nanmedian(X, axis=0)
        mask = np.isnan(X)
        X_clean = X.copy()
        # Correction appliquée ici aussi avec np.where !
        X_clean = np.where(mask, col_medians, X_clean)
    
    elif strategy == "drop":
        valid_rows = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_rows]
        if y is not None:
            y_clean = y[valid_rows]
            return X_clean, y_clean
        return X_clean
    
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'mean', 'median', or 'drop'.")
    
    if y is not None:
        return X_clean, y
    
    return X_clean

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = None) -> tuple:
    """
    Splits data into train and test sets.
    
    Parameters:
    -----------
    X : np.ndarray
        Features (m, d)
    y : np.ndarray
        Targets (m,)
    test_size : float
        Proportion of test set (0 to 1)
    random_state : int
        Seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[0]
    test_count = int(m * test_size)
    
    indices = np.random.permutation(m)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices])


def remove_outliers(X: np.ndarray, y: np.ndarray = None, method: str = "iqr", 
                   threshold: float = 1.5) -> tuple:
    """
    Removes outliers from the dataset.
    
    Parameters:
    -----------
    X : np.ndarray
        Features (m, d)
    y : np.ndarray, optional
        Targets (m,)
    method : str
        'iqr': Interquartile range method
        'zscore': Z-score method
    threshold : float
        IQR multiplier (1.5) or Z-score threshold (3.0)
    
    Returns:
    --------
    X_clean, y_clean
    """
    if method == "iqr":
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
    
    elif method == "zscore":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        z_scores = np.abs((X - mean) / std)
        mask = np.all(z_scores < threshold, axis=1)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'iqr' or 'zscore'.")
    
    X_clean = X[mask]
    
    if y is not None:
        y_clean = y[mask]
        return X_clean, y_clean
    
    return X_clean


def feature_importance_check(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Computes basic feature statistics for importance analysis.
    
    Returns:
    --------
    dict
        Correlation with target, variance, etc.
    """
    correlations = np.corrcoef(X.T, y)[:-1, -1]
    variances = np.var(X, axis=0)
    
    return {
        "correlations": correlations,
        "variances": variances,
        "abs_correlations": np.abs(correlations),
        "feature_indices_by_importance": np.argsort(np.abs(correlations))[::-1]
    }
