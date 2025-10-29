"""
Evaluation module for model testing and analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluator class.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to run on
            class_names: List of class names
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        self.model.eval()
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device).squeeze()
                
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                self.predictions.extend(predicted.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        self.predictions = np.array(self.predictions)
        self.true_labels = np.array(self.true_labels)
        self.probabilities = np.array(self.probabilities)
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.2f}%")
        
        return metrics
    
    def compute_metrics(self) -> Dict:
        """
        Compute evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Accuracy
        accuracy = accuracy_score(self.true_labels, self.predictions) * 100
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def print_classification_report(self):
        """Print detailed classification report."""
        report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            digits=4
        )
        logger.info(f"\nClassification Report:\n{report}")
        print(report)
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save plot
            normalize: Whether to normalize confusion matrix
        """
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_per_class_metrics(self, save_path: Optional[str] = None):
        """
        Plot per-class precision, recall, and F1 scores.
        
        Args:
            save_path: Path to save plot
        """
        metrics = self.compute_metrics()
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, metrics['precision_per_class'], width, label='Precision')
        ax.bar(x, metrics['recall_per_class'], width, label='Recall')
        ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved per-class metrics to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Dummy model and data
    model = nn.Linear(10, 3)
    test_data = [(torch.randn(16, 10), torch.randint(0, 3, (16,))) for _ in range(5)]
    test_loader = DataLoader(test_data, batch_size=16)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device="cpu",
        class_names=['Class A', 'Class B', 'Class C']
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    print(f"Metrics: {metrics}")
