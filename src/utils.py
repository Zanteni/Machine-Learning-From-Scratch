"""
src/utils.py
Utility functions for data loading, path management, and result logging.
"""
import numpy as np
import os

def load_data(filepath: str, has_header: bool = True) -> tuple:
    """
    Loads the Breast Cancer CSV data.
    - Drops the 'id' column (col 0)
    - Encodes 'diagnosis' (col 1): M -> 1 (Malicious), B -> 0 (Benign)
    - Returns (X, y)
    """
    # On charge toutes les données en tant que chaînes de caractères (str) pour gérer le 'M' et 'B'
    raw_data = np.genfromtxt(filepath, delimiter=',', dtype=str, skip_header=1 if has_header else 0)
    
    # 1. Extraction et encodage de la cible y (colonne index 1 : 'diagnosis')
    diagnosis_col = raw_data[:, 1]
    y = np.where(diagnosis_col == 'M', 1, 0).astype(float)
    
    # 2. Extraction des caractéristiques X (on supprime l'id (0) et la target (1))
    X_str = raw_data[:, 2:]
    
    # Sécurité : Si une colonne vide s'est glissée à la fin à cause d'une virgule traînante
    if X_str.shape[1] > 0 and np.all(X_str[:, -1] == ''):
        X_str = X_str[:, :-1]
        
    X = X_str.astype(float)
    
    return X, y

def ensure_directory(path: str):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_metrics_to_csv(metrics: dict, filepath: str):
    """Saves evaluation metrics dictionary to a clean CSV file."""
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        ensure_directory(parent_dir)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Metric,Value\n")
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                value_str = ";".join(map(str, value.flatten()))
            else:
                value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            f.write(f"{key},{value_str}\n")