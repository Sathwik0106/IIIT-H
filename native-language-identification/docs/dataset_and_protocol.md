# Dataset and Experimental Protocol

## Dataset: IndicAccentDb

### Overview

The **IndicAccentDb** dataset from HuggingFace contains English speech recordings from Indian speakers with various native language backgrounds. This dataset is specifically designed for accent detection and native language identification research.

**HuggingFace Link**: https://huggingface.co/datasets/DarshanaS/IndicAccentDb

### Dataset Characteristics

- **Languages Covered**: Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, and others
- **Speaker Demographics**: 
  - Age groups: Adults and children
  - Gender: Balanced representation
  - Geographic regions: Various states across India
- **Speech Types**:
  - Word-level utterances
  - Sentence-level utterances
  - Spontaneous speech
- **Audio Quality**:
  - Sample rate: 16kHz (standard)
  - Format: WAV/MP3
  - Recording conditions: Controlled and in-the-wild

### Data Splits

We use the following data split strategy:

```
Training set:   70% of data
Validation set: 15% of data
Test set:       15% of data
```

Splits are stratified by native language to ensure balanced representation across all sets.

## Experimental Protocol

### 1. Feature Extraction

#### MFCC Features (Baseline)

- **Number of coefficients**: 13-20
- **Frame length**: 2048 samples
- **Hop length**: 512 samples
- **Mel bands**: 128
- **Deltas**: First and second-order derivatives included
- **Total features**: 39-60 dimensions

#### HuBERT Features (Self-Supervised)

- **Model**: `facebook/hubert-base-ls960`
- **Hidden size**: 768 dimensions
- **Layers analyzed**: All 12 transformer layers
- **Pooling methods**: Mean, max, and attention pooling
- **Feature normalization**: L2 normalization applied

### 2. Model Architectures

#### CNN Classifier

```
Input → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
→ Flatten → Dense(256, dropout=0.5) → Dense(128, dropout=0.3) → Output
```

#### BiLSTM Classifier

```
Input → BiLSTM(128) → BiLSTM(64) → Dense(128, dropout=0.3) → Output
```

#### Transformer Classifier

```
Input → Projection(256) → PositionalEncoding
→ TransformerEncoder(4 heads, 2 layers) → GlobalAvgPool
→ Dense(128) → Output
```

### 3. Training Configuration

**Hyperparameters**:
- Optimizer: Adam / AdamW
- Learning rate: 0.001 (with scheduler)
- Batch size: 16-64 (depends on architecture)
- Epochs: 50-100 (with early stopping)
- Loss function: Cross-entropy
- Regularization: Dropout, L2 weight decay

**Learning Rate Schedule**:
- Reduce on plateau (patience: 5 epochs, factor: 0.5)
- OR Cosine annealing with warmup

**Early Stopping**:
- Patience: 15-20 epochs
- Monitor: Validation accuracy or loss

### 4. Evaluation Metrics

**Primary Metrics**:
- Accuracy (overall and per-class)
- Precision (weighted and per-class)
- Recall (weighted and per-class)
- F1-Score (weighted and per-class)
- Confusion Matrix

**Analysis Metrics**:
- Layer-wise separability (Fisher criterion)
- Cross-age generalization accuracy
- Word vs. sentence level performance

### 5. Experimental Studies

#### Study 1: MFCC vs. HuBERT Comparison

**Objective**: Compare traditional acoustic features with self-supervised representations.

**Protocol**:
1. Train identical classifier architecture on MFCC features
2. Train same classifier on HuBERT features (layer 12)
3. Compare performance metrics
4. Analyze which features capture accent better

#### Study 2: HuBERT Layer-wise Analysis

**Objective**: Identify which HuBERT layer best captures accent information.

**Protocol**:
1. Extract features from all 12 layers
2. Train separate classifiers on each layer's output
3. Compute Fisher criterion for class separability
4. Rank layers by classification performance
5. Visualize layer progression

**Expected Findings**:
- Middle to upper layers (6-10) should capture more accent information
- Lower layers focus on phonetic details
- Upper layers capture higher-level linguistic patterns

#### Study 3: Cross-Age Generalization

**Objective**: Test if models trained on adult speech generalize to children.

**Protocol**:
1. Train model on adult speech only
2. Test on adult speech (in-domain performance)
3. Test on children's speech (cross-age performance)
4. Analyze performance degradation
5. Compare MFCC vs. HuBERT robustness

**Research Questions**:
- Do accent patterns remain consistent across ages?
- Which features are more age-invariant?
- What acoustic differences exist between adult and child speech?

#### Study 4: Word-Level vs. Sentence-Level

**Objective**: Compare accent detection at different linguistic units.

**Protocol**:
1. Filter dataset by speech level (word/sentence)
2. Train separate models for each level
3. Compare performance metrics
4. Analyze which level provides stronger accent cues

**Hypotheses**:
- Sentence-level should provide more context and better performance
- Word-level focuses on specific phoneme realizations
- Longer utterances improve accent detection accuracy

### 6. Ablation Studies

**Components to Ablate**:
- Feature normalization (with/without)
- Data augmentation techniques
- Model architecture components (attention, pooling methods)
- Number of training samples
- Duration of audio segments

### 7. Error Analysis

**Analyses to Perform**:
1. Confusion between similar languages (e.g., Tamil-Telugu)
2. Impact of audio quality on performance
3. Speaker-specific variability
4. Impact of English proficiency level
5. Misclassification patterns

### 8. Statistical Significance

**Tests to Apply**:
- Paired t-tests for model comparisons
- McNemar's test for classifier agreement
- Confidence intervals for metrics (bootstrap)
- Statistical power analysis

### 9. Reproducibility

**Measures Taken**:
- Fixed random seeds (42)
- Deterministic algorithms where possible
- Version pinning for all libraries
- Configuration files for all experiments
- Detailed logging of hyperparameters

### 10. Computational Resources

**Hardware Requirements**:
- GPU: NVIDIA GPU with 8GB+ VRAM (for HuBERT)
- RAM: 16GB+ system memory
- Storage: 50GB+ for dataset and models

**Training Time Estimates**:
- MFCC baseline: 30-60 minutes
- HuBERT fine-tuning: 2-4 hours
- Layer-wise analysis: 4-8 hours

## Ethical Considerations

1. **Privacy**: Ensure speaker consent and anonymization
2. **Bias**: Check for regional and demographic biases
3. **Fairness**: Equal performance across all language groups
4. **Use Cases**: Ethical applications only (no discrimination)
5. **Transparency**: Clear documentation of limitations

## Limitations

1. Dataset may not cover all Indian languages
2. Recording conditions may vary
3. Limited speaker diversity in some language groups
4. Accent detection != language proficiency
5. Model may not generalize to other English varieties (British, American)

## Future Directions

1. Expand to more Indian languages
2. Include other English varieties (L1 interference patterns)
3. Multi-modal analysis (audio + text)
4. Real-time inference optimization
5. Fairness and bias mitigation strategies
6. Cross-corpus evaluation
