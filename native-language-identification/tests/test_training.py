"""Tests for training module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_simple_training_loop():
    """Test a simple training loop."""
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)
    
    # Create model
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Train for one epoch
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))


if __name__ == "__main__":
    pytest.main([__file__])
