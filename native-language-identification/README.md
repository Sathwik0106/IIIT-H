# Native Language Identification of Indian English Speakers Using HuBERT

## 📋 Project Overview

This project develops a system to identify the native language (L1) of Indian speakers speaking English by analyzing accent patterns in their speech. The system leverages both traditional acoustic features (MFCCs) and self-supervised representations (HuBERT embeddings) to classify speakers' native languages.

### Key Features

- **Multi-Feature Approach**: Compares MFCC and HuBERT embeddings for accent detection
- **Layer-wise Analysis**: Analyzes which HuBERT layers best capture accent information
- **Multiple Architectures**: Supports CNN, BiLSTM, and Transformer models
- **Cross-Age Generalization**: Tests model performance across adult and child speakers
- **Multi-Level Analysis**: Compares word-level vs. sentence-level accent detection
- **Real-World Application**: Accent-aware cuisine recommendation system for restaurants

### Supported Languages

- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada
- Bengali
- Gujarati
- Marathi

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd native-language-identification

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Dataset

The project uses the [IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb) dataset from HuggingFace, which contains English speech recordings from Indian speakers with various native language backgrounds.

### Basic Usage

```python
from src.data import IndicAccentDataLoader, AudioPreprocessor
from src.features import MFCCExtractor, HuBERTFeatureExtractor
from src.models import create_classifier
from src.utils import load_config

# Load configuration
config = load_config('configs/default.yaml')

# Load dataset
loader = IndicAccentDataLoader()
dataset = loader.load_dataset()
splits = loader.create_splits()

# Extract features and train model
# See notebooks for detailed examples
```

## 📁 Project Structure

```
native-language-identification/
├── configs/                  # Configuration files
│   ├── default.yaml         # Default configuration
│   ├── hubert.yaml          # HuBERT-specific config
│   └── mfcc_baseline.yaml   # MFCC baseline config
├── data/                     # Data directory
│   ├── raw/                 # Raw dataset
│   ├── processed/           # Processed features
│   └── metadata/            # Dataset metadata
├── src/                      # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature extraction
│   ├── models/              # Model architectures
│   ├── training/            # Training utilities
│   ├── evaluation/          # Evaluation metrics
│   ├── application/         # Application code
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks
├── experiments/             # Experiment results
├── scripts/                 # Shell scripts
├── models/                  # Saved models
│   └── checkpoints/        # Model checkpoints
├── docs/                    # Documentation
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## 🔬 Experiments

### 1. MFCC Baseline

```bash
python scripts/train.sh --config configs/mfcc_baseline.yaml
```

### 2. HuBERT Layer-wise Analysis

```bash
python scripts/train.sh --config configs/hubert.yaml
```

### 3. Cross-Age Generalization

Train on adults, test on children to analyze age generalization.

### 4. Word vs. Sentence Level

Compare accent detection performance at different linguistic levels.

## 📊 Model Architectures

### CNN Classifier
- Convolutional layers for temporal pattern extraction
- Max pooling for feature aggregation
- Fully connected layers for classification

### BiLSTM Classifier
- Bidirectional LSTM for sequence modeling
- Captures temporal dependencies in speech
- Dense layers for final classification

### Transformer Classifier
- Multi-head self-attention mechanism
- Positional encoding for sequence information
- Feedforward networks for transformation

## 🍽️ Application: Accent-Aware Cuisine Recommender

The system includes a real-world application that recommends regional Indian cuisines based on detected accents:

```python
from src.application import CuisineRecommender, RestaurantApplication

# Initialize recommender
recommender = CuisineRecommender(
    model=trained_model,
    label_mapping=label_map,
    cuisine_mapping=cuisine_map,
    preprocessor=preprocessor,
    feature_extractor=feature_extractor
)

# Get recommendations
audio = load_customer_speech()
recommendations = recommender.recommend_cuisines(audio, top_k=3)
print(recommendations['recommended_cuisines'])
```

### Example Recommendations

- **Malayalam accent** → Appam, Puttu, Avial, Fish Moilee
- **Hindi accent** → Butter Chicken, Chole Bhature, Dal Makhani
- **Tamil accent** → Dosa, Idli, Sambar, Chettinad Chicken

## 📈 Performance Metrics

The system evaluates models using:
- Accuracy
- Precision, Recall, F1-Score (weighted and per-class)
- Confusion Matrix
- Layer-wise separability analysis (for HuBERT)

## 🔧 Configuration

Edit `configs/default.yaml` to customize:
- Model architecture and hyperparameters
- Feature extraction settings
- Training parameters
- Data split ratios
- Evaluation metrics

## 📝 Notebooks

1. `01_data_exploration.ipynb` - Dataset analysis and visualization
2. `02_feature_extraction.ipynb` - MFCC and HuBERT feature extraction
3. `03_training_and_eval.ipynb` - Model training and evaluation

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_data.py
```

## 📚 Documentation

For detailed information about the dataset and experimental protocol, see `docs/dataset_and_protocol.md`.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)
- HuBERT Model: [Facebook AI Research](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- Transformers Library: [Hugging Face](https://huggingface.co/transformers/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🎯 Future Work

- Extend to more Indian languages
- Real-time accent detection
- Mobile application deployment
- Multi-modal analysis (audio + text)
- Cross-lingual accent transfer studies

---

**Note**: This is an academic/research project for native language identification. Ensure ethical use and respect for linguistic diversity.
