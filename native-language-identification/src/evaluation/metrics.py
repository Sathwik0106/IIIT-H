"""Evaluation metrics utilities."""

import numpy as np
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return accuracy_score(y_true, y_pred) * 100


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """Compute precision."""
    return precision_score(y_true, y_pred, average=average, zero_division=0) * 100


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """Compute recall."""
    return recall_score(y_true, y_pred, average=average, zero_division=0) * 100


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
    """Compute F1 score."""
    return f1_score(y_true, y_pred, average=average, zero_division=0) * 100


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'f1_score': compute_f1(y_true, y_pred)
    }
