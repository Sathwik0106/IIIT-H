"""
HuBERT fine-tuning module.
"""

import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig

logger = logging.getLogger(__name__)


class HuBERTForClassification(nn.Module):
    """
    HuBERT model with classification head.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        num_classes: int = 8,
        freeze_encoder: bool = False,
        unfreeze_layers: int = 0,
        pooling: str = "mean",
        dropout: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize HuBERT classification model.
        
        Args:
            model_name: Pre-trained HuBERT model name
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the encoder
            unfreeze_layers: Number of top layers to unfreeze (if freeze_encoder=True)
            pooling: Pooling method ('mean', 'max', 'attention')
            dropout: Dropout rate for classification head
            cache_dir: Cache directory for model
        """
        super(HuBERTForClassification, self).__init__()
        
        # Load pre-trained HuBERT
        self.hubert = HubertModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.hidden_size = self.hubert.config.hidden_size
        self.pooling = pooling
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.hubert.parameters():
                param.requires_grad = False
            
            # Unfreeze top layers
            if unfreeze_layers > 0:
                num_layers = len(self.hubert.encoder.layers)
                for layer in self.hubert.encoder.layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Attention pooling
        if pooling == "attention":
            self.attention = nn.Linear(self.hidden_size, 1)
        
        logger.info(f"Initialized HuBERT classifier with {num_classes} classes")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_values: Input audio tensor
            attention_mask: Attention mask
            
        Returns:
            Classification logits
        """
        # Extract features
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)
        
        # Apply pooling
        if self.pooling == "mean":
            pooled = torch.mean(hidden_states, dim=1)
        elif self.pooling == "max":
            pooled = torch.max(hidden_states, dim=1)[0]
        elif self.pooling == "attention":
            # Attention pooling
            attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
            pooled = torch.sum(hidden_states * attention_weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    model = HuBERTForClassification(
        model_name="facebook/hubert-base-ls960",
        num_classes=8,
        freeze_encoder=True,
        unfreeze_layers=2
    )
    
    # Dummy input
    batch_size = 2
    seq_length = 16000 * 3  # 3 seconds
    x = torch.randn(batch_size, seq_length)
    
    output = model(x)
    print(f"Output shape: {output.shape}")
