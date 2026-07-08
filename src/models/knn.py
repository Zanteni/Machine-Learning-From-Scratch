"""
models/knn.py
K-Nearest Neighbors classifier supporting multiple metrics and distance weighting.
"""
import numpy as np

class KNN:
    def __init__(self, k: int = 5, metric: str = "euclidean", p: float = 3.0, weighted: bool = False):
        self.k = k
        self.metric = metric
        self.p = p
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y.ravel()

    def _compute_distances(self, x_test: np.ndarray) -> np.ndarray:
        """
        Computes distances between one test vector (d,) and all training samples (n, d).
        Returns a 1D array of shape (n,).
        """
        error = x_test - self.X_train  # Matrix broadcasting: shape (n, d)
        
        if self.metric == "euclidean":
            return np.sqrt(np.sum(error**2, axis=1))
            
        elif self.metric == "manhattan":
            return np.sum(np.abs(error), axis=1)
            
        elif self.metric == "minkowski":
            return np.sum(np.abs(error)**self.p, axis=1) ** (1.0 / self.p)
            
        elif self.metric == "chebyshev":
            return np.max(np.abs(error), axis=1)
            
        else:
            raise ValueError(f"Unknown metric='{self.metric}'")

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        
        for x_test in X:
            # Step 1 & 2: Compute global distances and get sorted indices
            distances = self._compute_distances(x_test)
            sorted_indices = np.argsort(distances)
            
            # Step 3: Extract top k neighbor indices and their target classes
            first_k_indices = sorted_indices[:self.k]
            neighbor_classes = self.y_train[first_k_indices]
            
            # Step 4: Voting Phase
            if self.weighted:
                # Calculate inverse distance weights to give closer neighbors more influence
                first_k_distances = distances[first_k_indices]
                weights = 1.0 / (first_k_distances + 1e-5)
                
                # Accumulate weights for each unique class present in the neighborhood
                unique_classes = np.unique(neighbor_classes)
                class_scores = {c: np.sum(weights[neighbor_classes == c]) for c in unique_classes}
                
                pred_class = max(class_scores, key=class_scores.get)
            else:
                # Majority vote: select the most frequent class label among neighbors
                pred_class = np.bincount(neighbor_classes).argmax()
                
            predictions.append(pred_class)
            
        return np.array(predictions)