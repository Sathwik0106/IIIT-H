"""
Neural network classifiers for native language identification.
Includes CNN, BiLSTM, and Transformer architectures.
"""

import logging
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CNNClassifier(nn.Module):
    """
    CNN-based classifier for accent classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_layers: List[Dict],
        dense_layers: List[Dict],
        input_is_sequence: bool = True,
        batch_norm: bool = True
    ):
        """
        Initialize CNN classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            conv_layers: List of conv layer configs with 'filters', 'kernel_size', 'pool_size'
            dense_layers: List of dense layer configs with 'units', 'dropout'
            input_is_sequence: Whether input is a sequence (True) or fixed vector (False)
            batch_norm: Whether to use batch normalization
        """
        super(CNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.input_is_sequence = input_is_sequence
        self.batch_norm_enabled = batch_norm
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = 1  # Single channel input
        
        for i, layer_config in enumerate(conv_layers):
            filters = layer_config['filters']
            kernel_size = layer_config['kernel_size']
            pool_size = layer_config.get('pool_size', 2)
            activation = layer_config.get('activation', 'relu')
            
            block = nn.ModuleDict({
                'conv': nn.Conv2d(
                    in_channels,
                    filters,
                    kernel_size=kernel_size,
                    padding='same'
                ),
                'pool': nn.MaxPool2d(pool_size)
            })
            
            if batch_norm:
                block['bn'] = nn.BatchNorm2d(filters)
            
            self.conv_blocks.append(block)
            in_channels = filters
        
        # Calculate size after convolutions
        # This is approximate and depends on input size
        self.conv_output_size = None  # Will be computed in forward pass
        
        # Build dense layers
        self.dense_layers = nn.ModuleList()
        prev_units = None  # Will be set after first forward pass
        
        for i, layer_config in enumerate(dense_layers):
            units = layer_config['units']
            dropout = layer_config.get('dropout', 0.0)
            
            # First dense layer size will be determined dynamically
            if i == 0:
                self.first_dense_units = units
                self.dense_layers.append(nn.ModuleDict({
                    'linear': None,  # Placeholder
                    'dropout': nn.Dropout(dropout) if dropout > 0 else None
                }))
            else:
                self.dense_layers.append(nn.ModuleDict({
                    'linear': nn.Linear(prev_units, units),
                    'dropout': nn.Dropout(dropout) if dropout > 0 else None
                }))
            
            prev_units = units
        
        # Output layer
        self.output_layer = nn.Linear(prev_units, num_classes)
        
        logger.info(f"Initialized CNN classifier with {len(conv_layers)} conv layers "
                   f"and {len(dense_layers)} dense layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, features, time) or (batch, features)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape input for CNN if needed
        if x.dim() == 2:
            # Input is (batch, features) - add channel and time dimensions
            x = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, features, 1)
        elif x.dim() == 3:
            # Input is (batch, features, time) - add channel dimension
            x = x.unsqueeze(1)  # (batch, 1, features, time)
        
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block['conv'](x)
            if 'bn' in block and self.batch_norm_enabled:
                x = block['bn'](x)
            x = F.relu(x)
            x = block['pool'](x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Initialize first dense layer if needed
        if self.dense_layers[0]['linear'] is None:
            self.conv_output_size = x.size(1)
            self.dense_layers[0]['linear'] = nn.Linear(
                self.conv_output_size,
                self.first_dense_units
            ).to(x.device)
        
        # Apply dense layers
        for block in self.dense_layers:
            x = block['linear'](x)
            x = F.relu(x)
            if block['dropout'] is not None:
                x = block['dropout'](x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM-based classifier for accent classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lstm_units: List[int],
        dropout: float = 0.3,
        recurrent_dropout: float = 0.0,
        dense_units: int = 128
    ):
        """
        Initialize BiLSTM classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            lstm_units: List of LSTM hidden units for each layer
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            dense_units: Number of units in final dense layer
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_size = input_dim
        
        for i, units in enumerate(lstm_units):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=prev_size,
                    hidden_size=units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0  # Will use separate dropout layer
                )
            )
            prev_size = units * 2  # Bidirectional doubles the output
        
        self.dropout = nn.Dropout(dropout)
        
        # Dense layer
        self.dense = nn.Linear(prev_size, dense_units)
        self.dense_dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(dense_units, num_classes)
        
        logger.info(f"Initialized BiLSTM classifier with {len(lstm_units)} LSTM layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, features) or (batch, features)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Ensure input is 3D: (batch, time, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dimension
        elif x.dim() == 3 and x.size(2) == self.input_dim:
            # Input is (batch, features, time) - transpose
            x = x.transpose(1, 2)
        
        # Apply LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        # Take last time step output
        x = x[:, -1, :]
        
        # Dense layer
        x = self.dense(x)
        x = F.relu(x)
        x = self.dense_dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for accent classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        dense_units: int = 128,
        max_seq_length: int = 1000,
        use_positional_encoding: bool = True
    ):
        """
        Initialize Transformer classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension
            num_layers: Number of transformer layers
            dropout: Dropout rate
            dense_units: Number of units in final dense layer
            max_seq_length: Maximum sequence length
            use_positional_encoding: Whether to use positional encoding
        """
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, ff_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(ff_dim, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(ff_dim, dense_units)
        self.output_layer = nn.Linear(dense_units, num_classes)
        
        logger.info(f"Initialized Transformer classifier with {num_layers} layers "
                   f"and {num_heads} attention heads")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, features) or (batch, features)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Ensure input is 3D: (batch, time, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dimension
        elif x.dim() == 3 and x.size(2) == self.input_dim:
            # Input is (batch, features, time) - transpose
            x = x.transpose(1, 2)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.dropout(x)
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


def create_classifier(
    architecture: str,
    input_dim: int,
    num_classes: int,
    config: Dict
) -> nn.Module:
    """
    Factory function to create classifier.
    
    Args:
        architecture: Architecture type ('cnn', 'bilstm', 'transformer')
        input_dim: Input feature dimension
        num_classes: Number of output classes
        config: Architecture-specific configuration
        
    Returns:
        Classifier model
    """
    if architecture.lower() == 'cnn':
        return CNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **config.get('cnn', {})
        )
    
    elif architecture.lower() == 'bilstm':
        return BiLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **config.get('bilstm', {})
        )
    
    elif architecture.lower() == 'transformer':
        return TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **config.get('transformer', {})
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test CNN
    cnn_config = {
        'cnn': {
            'conv_layers': [
                {'filters': 32, 'kernel_size': 3, 'pool_size': 2},
                {'filters': 64, 'kernel_size': 3, 'pool_size': 2}
            ],
            'dense_layers': [
                {'units': 128, 'dropout': 0.5}
            ]
        }
    }
    
    cnn = create_classifier('cnn', input_dim=40, num_classes=8, config=cnn_config)
    x = torch.randn(4, 40, 100)  # (batch, features, time)
    output = cnn(x)
    print(f"CNN output shape: {output.shape}")
    
    # Test BiLSTM
    bilstm_config = {
        'bilstm': {
            'lstm_units': [128, 64],
            'dropout': 0.3,
            'dense_units': 128
        }
    }
    
    bilstm = create_classifier('bilstm', input_dim=40, num_classes=8, config=bilstm_config)
    output = bilstm(x)
    print(f"BiLSTM output shape: {output.shape}")
    
    # Test Transformer
    transformer_config = {
        'transformer': {
            'num_heads': 4,
            'ff_dim': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'dense_units': 128
        }
    }
    
    transformer = create_classifier('transformer', input_dim=40, num_classes=8, config=transformer_config)
    output = transformer(x)
    print(f"Transformer output shape: {output.shape}")
