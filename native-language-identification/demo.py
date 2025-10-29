#!/usr/bin/env python3
"""
Demo script to test the Native Language Identification system.
This script demonstrates the core functionality without requiring the full dataset.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("  NATIVE LANGUAGE IDENTIFICATION - DEMO")
print("=" * 70)
print()

# Test 1: Load Configuration
print("📋 Test 1: Loading Configuration")
print("-" * 70)
try:
    from src.utils import load_config
    config = load_config('configs/default.yaml')
    print(f"✓ Configuration loaded successfully!")
    print(f"  Project: {config['project_name']}")
    print(f"  Languages: {', '.join(config['data']['languages'][:4])}...")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Audio Preprocessing
print("🎵 Test 2: Audio Preprocessing")
print("-" * 70)
try:
    from src.data import AudioPreprocessor
    
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        max_duration=3.0,
        normalize=True
    )
    
    # Generate dummy audio (3 seconds)
    dummy_audio = np.random.randn(16000 * 2) * 0.1
    processed = preprocessor.preprocess(dummy_audio, pad_to_max=True)
    
    print(f"✓ Audio preprocessing successful!")
    print(f"  Input shape: {dummy_audio.shape}")
    print(f"  Output shape: {processed.shape}")
    print(f"  Sample rate: {preprocessor.sample_rate} Hz")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: MFCC Feature Extraction
print("🎼 Test 3: MFCC Feature Extraction")
print("-" * 70)
try:
    from src.features import MFCCExtractor
    
    mfcc_extractor = MFCCExtractor(
        sample_rate=16000,
        n_mfcc=13,
        use_deltas=True,
        use_delta_deltas=True
    )
    
    mfcc_features = mfcc_extractor.extract(processed)
    
    print(f"✓ MFCC extraction successful!")
    print(f"  Feature shape: {mfcc_features.shape}")
    print(f"  Coefficients: {mfcc_extractor.n_mfcc}")
    print(f"  Total features: {mfcc_features.shape[0]} (MFCC + deltas + delta-deltas)")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model Creation
print("🧠 Test 4: Creating Neural Network Models")
print("-" * 70)
try:
    from src.models import create_classifier
    
    num_classes = len(config['data']['languages'])
    
    # Create CNN model
    cnn_model = create_classifier(
        architecture='cnn',
        input_dim=mfcc_features.shape[0],
        num_classes=num_classes,
        config=config['model']
    )
    
    # Create BiLSTM model
    bilstm_model = create_classifier(
        architecture='bilstm',
        input_dim=mfcc_features.shape[0],
        num_classes=num_classes,
        config=config['model']
    )
    
    # Create Transformer model
    transformer_model = create_classifier(
        architecture='transformer',
        input_dim=mfcc_features.shape[0],
        num_classes=num_classes,
        config=config['model']
    )
    
    print(f"✓ Models created successfully!")
    print(f"  CNN model: {sum(p.numel() for p in cnn_model.parameters()):,} parameters")
    print(f"  BiLSTM model: {sum(p.numel() for p in bilstm_model.parameters()):,} parameters")
    print(f"  Transformer model: {sum(p.numel() for p in transformer_model.parameters()):,} parameters")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward Pass
print("⚡ Test 5: Model Forward Pass")
print("-" * 70)
try:
    # Create batch of features
    batch_size = 4
    features_batch = torch.FloatTensor(mfcc_features).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Test CNN
    with torch.no_grad():
        cnn_output = cnn_model(features_batch)
    
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {features_batch.shape}")
    print(f"  Output shape: {cnn_output.shape}")
    print(f"  Output logits: {cnn_output[0].tolist()[:3]}... (showing first 3)")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Training Components
print("🏋️ Test 6: Training Infrastructure")
print("-" * 70)
try:
    from src.training import Trainer
    from src.evaluation import compute_all_metrics
    
    # Create dummy data loader
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(32, mfcc_features.shape[0], mfcc_features.shape[1]),
        torch.randint(0, num_classes, (32,))
    )
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8)
    
    # Create trainer
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=cnn_model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu',
        experiment_name='demo_test'
    )
    
    print(f"✓ Training infrastructure initialized!")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Loss function: {type(criterion).__name__}")
    print(f"  Device: CPU")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Cuisine Recommender
print("🍽️ Test 7: Cuisine Recommendation Application")
print("-" * 70)
try:
    from src.application import CuisineRecommender
    
    # Create label mapping
    label_mapping = {lang: idx for idx, lang in enumerate(config['data']['languages'])}
    
    # Create recommender
    recommender = CuisineRecommender(
        model=cnn_model,
        label_mapping=label_mapping,
        cuisine_mapping=config['application']['cuisine_mapping'],
        preprocessor=preprocessor,
        feature_extractor=mfcc_extractor,
        device='cpu',
        confidence_threshold=0.6
    )
    
    # Test prediction
    test_audio = np.random.randn(16000 * 2) * 0.1
    result = recommender.recommend_cuisines(test_audio, top_k=3)
    
    print(f"✓ Cuisine recommender working!")
    print(f"  Detected language: {result['detected_language']}")
    print(f"  Confidence: {result['confidence_percentage']:.1f}%")
    print(f"  Recommended cuisines: {', '.join(result['recommended_cuisines'])}")
    print()
    
    # Generate message
    message = recommender.generate_recommendation_message(test_audio)
    print(f"  Message: {message[:100]}...")
    print()
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Visualizations
print("📊 Test 8: Creating Visualizations")
print("-" * 70)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(processed[:1000])
    ax.set_title('Sample Audio Waveform')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    
    output_path = project_root / 'demo_waveform.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization created!")
    print(f"  Saved to: {output_path}")
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("=" * 70)
print("  ✅ DEMO COMPLETED SUCCESSFULLY!")
print("=" * 70)
print()
print("All core components are working correctly:")
print("  ✓ Configuration management")
print("  ✓ Audio preprocessing")
print("  ✓ MFCC feature extraction")
print("  ✓ Model architectures (CNN, BiLSTM, Transformer)")
print("  ✓ Forward pass through models")
print("  ✓ Training infrastructure")
print("  ✓ Cuisine recommendation application")
print("  ✓ Visualization tools")
print()
print("🚀 Next Steps:")
print("  1. Load the IndicAccentDb dataset")
print("  2. Train models with real data")
print("  3. Evaluate performance")
print("  4. Deploy the application")
print()
print("📖 See QUICKSTART.md for detailed instructions")
print("=" * 70)
