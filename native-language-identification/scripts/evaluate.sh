#!/bin/bash

# Evaluation Script for Native Language Identification

set -e

# Default parameters
CONFIG_FILE="configs/default.yaml"
MODEL_PATH="models/checkpoints/best_model.pt"
OUTPUT_DIR="experiments/results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Evaluating Native Language Identification Model"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python -c "
import torch
from src.data import IndicAccentDataLoader, AudioPreprocessor, NLIDataset, create_dataloader
from src.features import MFCCExtractor
from src.models import create_classifier
from src.evaluation import Evaluator
from src.utils import load_config

# Load configuration
config = load_config('$CONFIG_FILE')

# Load dataset
loader = IndicAccentDataLoader()
dataset = loader.load_dataset()
splits = loader.create_splits()

# Create test dataset
preprocessor = AudioPreprocessor()
feature_extractor = MFCCExtractor()

test_dataset = NLIDataset(
    data=splits['test'],
    preprocessor=preprocessor,
    feature_extractor=feature_extractor,
    label_mapping=loader.label_mapping
)

test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = create_classifier(
    architecture=config['model']['architecture'],
    input_dim=40,
    num_classes=len(loader.label_mapping),
    config=config['model']
)

checkpoint = torch.load('$MODEL_PATH')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
evaluator = Evaluator(
    model=model,
    test_loader=test_loader,
    device='cuda',
    class_names=list(loader.label_mapping.keys())
)

metrics = evaluator.evaluate()
evaluator.print_classification_report()
evaluator.plot_confusion_matrix(save_path='$OUTPUT_DIR/confusion_matrix.png')
evaluator.plot_per_class_metrics(save_path='$OUTPUT_DIR/per_class_metrics.png')

print('Evaluation completed!')
print(f'Results saved to: $OUTPUT_DIR')
"

echo "Evaluation completed successfully!"
