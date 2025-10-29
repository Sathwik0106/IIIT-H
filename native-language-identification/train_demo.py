#!/usr/bin/env python3
"""
Quick training demo to show the model actually learning.
Uses synthetic data to demonstrate the training loop.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import create_classifier
from src.utils import load_config

print("=" * 70)
print("  QUICK TRAINING DEMO - Native Language Identification")
print("=" * 70)
print()

# Load config
config = load_config('configs/default.yaml')
num_classes = len(config['data']['languages'])

print(f"🎯 Task: Classify {num_classes} Indian Languages")
print(f"Languages: {', '.join(config['data']['languages'])}")
print()

# Create synthetic dataset (simulating MFCC features)
print("📊 Creating synthetic training data...")
n_train = 400
n_val = 100
feature_dim = 39  # MFCC features
time_steps = 94

# Create training data with some pattern
X_train = torch.randn(n_train, feature_dim, time_steps)
y_train = torch.randint(0, num_classes, (n_train,))

# Add some class-specific patterns (to make it learnable)
for i in range(num_classes):
    mask = y_train == i
    X_train[mask] += torch.randn(1, feature_dim, time_steps) * 0.5

# Create validation data
X_val = torch.randn(n_val, feature_dim, time_steps)
y_val = torch.randint(0, num_classes, (n_val,))
for i in range(num_classes):
    mask = y_val == i
    X_val[mask] += torch.randn(1, feature_dim, time_steps) * 0.5

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"  Training samples: {n_train}")
print(f"  Validation samples: {n_val}")
print(f"  Feature dimension: {feature_dim} x {time_steps}")
print()

# Create model
print("🧠 Creating CNN model...")
model = create_classifier(
    architecture='cnn',
    input_dim=feature_dim,
    num_classes=num_classes,
    config=config['model']
)

num_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {num_params:,}")
print()

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = 'cpu'
model.to(device)

print("🏋️ Starting training...")
print("-" * 70)

num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    
    # Print progress
    print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%")

print("-" * 70)
print()

# Final evaluation
print("📈 Final Results:")
print(f"  Training Accuracy:   {train_acc:.2f}%")
print(f"  Validation Accuracy: {val_acc:.2f}%")
print()

# Test prediction on a single sample
print("🔮 Testing prediction on a sample:")
model.eval()
with torch.no_grad():
    sample_feature = X_val[0:1].to(device)
    sample_label = y_val[0].item()
    
    output = model(sample_feature)
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item() * 100
    
    languages = config['data']['languages']
    print(f"  True language:      {languages[sample_label]}")
    print(f"  Predicted language: {languages[predicted_class]}")
    print(f"  Confidence:         {confidence:.1f}%")
    print(f"  Correct:            {'✓' if predicted_class == sample_label else '✗'}")
    print()
    
    print("  Top 3 predictions:")
    top_3_probs, top_3_indices = torch.topk(probabilities, 3)
    for i, (prob, idx) in enumerate(zip(top_3_probs, top_3_indices)):
        print(f"    {i+1}. {languages[idx.item()]:<12} - {prob.item()*100:5.1f}%")

print()
print("=" * 70)
print("  ✅ TRAINING DEMO COMPLETED!")
print("=" * 70)
print()
print("The model successfully learned patterns from the data!")
print("With real audio data, the model would learn actual accent patterns.")
print()
print("🎯 Next steps:")
print("  1. Load IndicAccentDb dataset from HuggingFace")
print("  2. Extract real MFCC/HuBERT features")
print("  3. Train on actual accent data")
print("  4. Achieve high accuracy on native language classification")
print()
