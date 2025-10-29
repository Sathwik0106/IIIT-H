"""Evaluation module initialization."""

from .evaluate import Evaluator
from .metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_all_metrics
)

__all__ = [
    'Evaluator',
    'compute_accuracy',
    'compute_precision',
    'compute_recall',
    'compute_f1',
    'compute_all_metrics'
]
