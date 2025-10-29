#!/bin/bash

# Training Script for Native Language Identification

set -e

# Default parameters
CONFIG_FILE="configs/default.yaml"
EXPERIMENT_NAME="experiment_$(date +%Y%m%d_%H%M%S)"
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Training Native Language Identification Model"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
fi
echo "=========================================="

# Run training
python -c "
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import IndicAccentDataLoader, AudioPreprocessor, NLIDataset, create_dataloader
from src.features import MFCCExtractor, HuBERTFeatureExtractor
from src.models import create_classifier
from src.training import Trainer
from src.utils import load_config, Config

# Load configuration
print('Loading configuration...')
config_dict = load_config('$CONFIG_FILE')
config = Config(config_dict)

# Set seed for reproducibility
torch.manual_seed(config.get('seed', 42))

# Load dataset
print('Loading dataset...')
loader = IndicAccentDataLoader(
    cache_dir=config['data']['cache_dir'],
    processed_dir=config['data']['processed_dir']
)
dataset = loader.load_dataset()
splits = loader.create_splits(
    train_size=config['data']['train_split'],
    val_size=config['data']['val_split'],
    test_size=config['data']['test_split'],
    seed=config.get('seed', 42)
)

# Create preprocessor and feature extractor
preprocessor = AudioPreprocessor(
    sample_rate=config['data']['sample_rate'],
    max_duration=config['data']['max_duration']
)

if config['features'].get('use_mfcc', True):
    feature_extractor = MFCCExtractor(**config['features']['mfcc'])
else:
    feature_extractor = HuBERTFeatureExtractor(**config['features']['hubert'])

# Create datasets
print('Creating data loaders...')
train_dataset = NLIDataset(
    data=splits['train'],
    preprocessor=preprocessor,
    feature_extractor=feature_extractor,
    label_mapping=loader.label_mapping,
    apply_augmentation=True
)

val_dataset = NLIDataset(
    data=splits['val'],
    preprocessor=preprocessor,
    feature_extractor=feature_extractor,
    label_mapping=loader.label_mapping,
    apply_augmentation=False
)

# Create data loaders
train_loader = create_dataloader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=config['hardware']['num_workers']
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['hardware']['num_workers']
)

# Create model
print('Creating model...')
num_classes = len(loader.label_mapping)
model = create_classifier(
    architecture=config['model']['architecture'],
    input_dim=40,  # This should be determined from features
    num_classes=num_classes,
    config=config['model']
)

# Create optimizer and criterion
optimizer = optim.Adam(
    model.parameters(),
    lr=config['training']['learning_rate']
)
criterion = nn.CrossEntropyLoss()

# Create trainer
print('Initializing trainer...')
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=config['hardware']['device'],
    checkpoint_dir=config['training']['checkpoint']['save_dir'],
    experiment_name='$EXPERIMENT_NAME'
)

# Load checkpoint if resuming
if '$RESUME':
    trainer.load_checkpoint('$RESUME')

# Train
print('Starting training...')
trainer.train(
    num_epochs=config['training']['epochs'],
    early_stopping_patience=config['training']['early_stopping']['patience'],
    save_best_only=config['training']['checkpoint']['save_best_only']
)

print('Training completed!')
"

echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="
