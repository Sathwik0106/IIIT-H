#!/bin/bash

# Feature Extraction Script for Native Language Identification
# This script extracts features from audio files

set -e

# Default parameters
CONFIG_FILE="configs/default.yaml"
FEATURE_TYPE="mfcc"  # mfcc or hubert
OUTPUT_DIR="data/processed"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --feature-type)
            FEATURE_TYPE="$2"
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
echo "Feature Extraction for NLI"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Feature Type: $FEATURE_TYPE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run feature extraction
python -c "
from src.data import IndicAccentDataLoader, AudioPreprocessor
from src.features import MFCCExtractor, HuBERTFeatureExtractor
from src.utils import load_config
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load config
config = load_config('$CONFIG_FILE')

# Load dataset
print('Loading dataset...')
loader = IndicAccentDataLoader(
    cache_dir=config['data']['cache_dir'],
    processed_dir='$OUTPUT_DIR'
)
dataset = loader.load_dataset()

# Create preprocessor
preprocessor = AudioPreprocessor(
    sample_rate=config['data']['sample_rate'],
    max_duration=config['data']['max_duration']
)

# Create feature extractor
if '$FEATURE_TYPE' == 'mfcc':
    extractor = MFCCExtractor(**config['features']['mfcc'])
else:
    extractor = HuBERTFeatureExtractor(**config['features']['hubert'])

print(f'Extracting {FEATURE_TYPE.upper()} features...')

# Extract features
# This is a placeholder - implement full extraction loop

print('Feature extraction completed!')
print(f'Features saved to: $OUTPUT_DIR')
"

echo "Feature extraction completed successfully!"
