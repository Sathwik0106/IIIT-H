# Quick Start Guide - Native Language Identification

## 🚀 Getting Started

### 1. Installation

```bash
# Navigate to project directory
cd native-language-identification

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Dataset Download

The project uses the IndicAccentDb dataset from HuggingFace. It will be automatically downloaded when you run the data loader for the first time.

```python
from src.data import IndicAccentDataLoader

loader = IndicAccentDataLoader()
dataset = loader.load_dataset()
```

### 3. Configuration

Edit `configs/default.yaml` to customize your experiment:

```yaml
# Example: Change model architecture
model:
  architecture: "cnn"  # Options: cnn, bilstm, transformer

# Example: Adjust training parameters
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
```

### 4. Quick Training

#### Option A: Using Jupyter Notebooks (Recommended for beginners)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### Option B: Using Shell Scripts

```bash
# Train MFCC baseline
./scripts/train.sh --config configs/mfcc_baseline.yaml

# Train HuBERT model
./scripts/train.sh --config configs/hubert.yaml

# Evaluate model
./scripts/evaluate.sh --model models/checkpoints/best_model.pt
```

#### Option C: Using Python Scripts

```python
from src.data import IndicAccentDataLoader, AudioPreprocessor, NLIDataset
from src.features import MFCCExtractor
from src.models import create_classifier
from src.training import Trainer
from src.utils import load_config
import torch.nn as nn
import torch.optim as optim

# Load config
config = load_config('configs/default.yaml')

# Load and prepare data
loader = IndicAccentDataLoader()
dataset = loader.load_dataset()
splits = loader.create_splits()

# Create preprocessor and feature extractor
preprocessor = AudioPreprocessor()
feature_extractor = MFCCExtractor()

# Create datasets
train_dataset = NLIDataset(
    data=splits['train'],
    preprocessor=preprocessor,
    feature_extractor=feature_extractor,
    label_mapping=loader.label_mapping
)

# Create model
model = create_classifier(
    architecture='cnn',
    input_dim=40,
    num_classes=len(loader.label_mapping),
    config=config['model']
)

# Train
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer
)

trainer.train(num_epochs=50)
```

## 📊 Experiments

### 1. MFCC Baseline

Train a CNN model with MFCC features:

```bash
./scripts/train.sh --config configs/mfcc_baseline.yaml --experiment-name mfcc_baseline
```

### 2. HuBERT Layer Analysis

Analyze which HuBERT layer captures accent best:

```bash
./scripts/train.sh --config configs/hubert.yaml --experiment-name hubert_layer_analysis
```

### 3. Cross-Age Generalization

Train on adults, test on children:

```python
# Filter datasets by age group
adult_train = loader.filter_by_age_group(splits['train'], 'adult')
child_test = loader.filter_by_age_group(splits['test'], 'child')

# Train and evaluate...
```

### 4. Word vs Sentence Level

```python
# Filter by speech level
word_data = loader.filter_by_speech_level(dataset, 'word')
sentence_data = loader.filter_by_speech_level(dataset, 'sentence')

# Train separate models and compare...
```

## 🍽️ Cuisine Recommendation Demo

```python
from src.application import CuisineRecommender

# Load trained model and create recommender
recommender = CuisineRecommender(
    model=trained_model,
    label_mapping=label_mapping,
    cuisine_mapping=cuisine_map,
    preprocessor=preprocessor,
    feature_extractor=feature_extractor
)

# Get recommendations from audio
audio = load_customer_audio()
result = recommender.recommend_cuisines(audio, top_k=3)

print(f"Detected: {result['detected_language']}")
print(f"Confidence: {result['confidence_percentage']:.1f}%")
print(f"Recommendations: {result['recommended_cuisines']}")
```

## 📈 Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

## 🔧 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Dataset download fails

**Solution**: Set HuggingFace cache directory:
```python
loader = IndicAccentDataLoader(cache_dir='/path/to/cache')
```

### Issue: ImportError for transformers

**Solution**: Install latest transformers:
```bash
pip install --upgrade transformers
```

## 📚 Additional Resources

- **Full Documentation**: `docs/dataset_and_protocol.md`
- **API Reference**: See docstrings in source code
- **Issues**: Report bugs on GitHub
- **Examples**: Check `notebooks/` directory

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Run data exploration notebook
3. ✅ Train baseline model
4. ✅ Experiment with HuBERT
5. ✅ Develop your application

## 💡 Tips

- Start with the MFCC baseline (faster training)
- Use smaller batch sizes for HuBERT (memory intensive)
- Monitor validation loss to prevent overfitting
- Save checkpoints frequently
- Use GPU for HuBERT training (much faster)

Happy coding! 🎉
