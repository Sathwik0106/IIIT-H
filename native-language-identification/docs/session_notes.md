# Session notes — Native Language Identification (HuBERT + MFCC)

Date: 2025-10-29

## Project goal
Identify the native language of Indian English speakers from speech (8 languages) using MFCC and HuBERT features, compare models (CNN/BiLSTM/Transformer), perform HuBERT layer-wise analysis, evaluate cross-age (adults→children) and unit (word vs sentence), and power a cuisine recommendation demo.

## Current status
- Full project scaffolded and pushed to GitHub under `native-language-identification/`
- Demos run successfully:
  - `demo.py` component tests all pass
  - `train_demo.py` shows the model learns on synthetic data (Train Acc → 100%)
- Dependencies installed; TensorBoard available
- Clean `.gitignore` to avoid large artifacts in repo

## Where to resume next
1) Load real dataset (IndicAccentDb) via HuggingFace
2) Extract MFCC and HuBERT features
3) Train MFCC baseline; then HuBERT; run layer-wise analysis
4) Cross-age eval (train adults → test children) and word vs sentence experiments
5) Connect predictions to Cuisine Recommender app

## Quick commands (next session)
Run from the project root:

```bash
# 1) Explore data and configs
cd /workspaces/IIIT-H/native-language-identification
python3 demo.py

# 2) Synthetic training demo (sanity check)
python3 train_demo.py

# 3) Train MFCC baseline (uses configs/mfcc_baseline.yaml)
./scripts/train.sh --config configs/mfcc_baseline.yaml

# 4) Train HuBERT model (uses configs/hubert.yaml)
./scripts/train.sh --config configs/hubert.yaml

# 5) Evaluate a saved checkpoint
./scripts/evaluate.sh --model models/checkpoints/best_model.pt
```

## Code pointers
- Configs: `configs/default.yaml`, `configs/mfcc_baseline.yaml`, `configs/hubert.yaml`
- Data pipeline: `src/data/loader.py`, `src/data/preprocessing.py`, `src/data/dataset.py`
- Features: `src/features/traditional.py` (MFCC), `src/features/hubert_features.py`
- Models: `src/models/classifier.py`, `src/models/hubert_finetune.py`
- Training: `src/training/train.py`, `src/training/callbacks.py`
- Evaluation: `src/evaluation/evaluate.py`
- App demo: `src/application/cuisine_recommender.py`

## Notes
- Large artifacts (data, logs, checkpoints) are gitignored; use `resources/pretrained/` for local models if needed.
- For HuBERT, the default is `facebook/hubert-base-ls960`. Consider caching models to avoid repeated downloads.
- TensorBoard logs are written under `logs/` (gitignored). Launch with `tensorboard --logdir logs` (optional).

## Resume this context in a new chat
Paste this prompt in a new chat:

```
Resume my NLI project using notes from native-language-identification/docs/session_notes.md. Start with loading IndicAccentDb and MFCC extraction, then train MFCC baseline.
```

You can also open the GitHub issue “NLI Project: Next steps” and check tasks as you complete them.