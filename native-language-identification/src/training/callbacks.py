"""Training callbacks."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if should stop.
        
        Args:
            current_value: Current metric value
            
        Returns:
            Whether to stop training
        """
        if self.mode == 'min':
            is_improvement = current_value < (self.best_value - self.min_delta)
        else:
            is_improvement = current_value > (self.best_value + self.min_delta)
        
        if is_improvement:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.should_stop


class ModelCheckpoint:
    """Model checkpoint callback."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        """
        Initialize model checkpoint.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to save only best model
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def is_better(self, current_value: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == 'min':
            return current_value < self.best_value
        else:
            return current_value > self.best_value
