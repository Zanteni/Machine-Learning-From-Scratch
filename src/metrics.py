"""
src/metrics.py
Evaluation metrics for classification tasks.
"""
import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates accuracy: (TP + TN) / Total
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)
    
    Returns:
    --------
    float
        Accuracy score in [0, 1]
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the binary confusion matrix.
    
    Returns:
    --------
    np.ndarray
        2x2 matrix: [[TN, FP], [FN, TP]]
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)
    (Of positive predictions, how many were correct?)
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    if TP + FP == 0:
        return 0.0
    
    return TP / (TP + FP)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN)
    (Of actual positives, how many were detected?)
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    if TP + FN == 0:
        return 0.0
    
    return TP / (TP + FN)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    Harmonic mean of precision and recall.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Specificity = TN / (TN + FP)
    (Of actual negatives, how many were correctly identified?)
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    if TN + FP == 0:
        return 0.0
    
    return TN / (TN + FP)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes all metrics at once and returns as a dictionary.
    
    Returns:
    --------
    dict
        Dictionary with keys: accuracy, precision, recall, f1, specificity, cm
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
