"""
src/visualization.py
Plotting utilities for EDA and results.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Plots a 2x2 confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    # Add text annotations with dynamic contrast
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)
            
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_loss_curves(history: dict, title: str = "Loss Over Epochs"):
    """Plots training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_model_comparison(metrics_dict: dict):
    """Compares metrics across models."""
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    f1_scores = [metrics_dict[m]['f1_score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y')
    plt.show()