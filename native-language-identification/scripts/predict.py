#!/usr/bin/env python3
"""
Predict native language from a single audio file.

Usage:
  python3 scripts/predict.py \
    --audio path/to/file.wav \
    [--config configs/default.yaml] \
    [--checkpoint models/checkpoints/best_model.pt] \
    [--features mfcc|hubert] \
    [--arch cnn|bilstm|transformer] \
    [--device auto|cpu|cuda]

Notes:
  - If no checkpoint is provided and none found, the model will be untrained
    and predictions will be essentially random. Train first for meaningful results.
  - Default feature pipeline uses MFCC.
"""

import argparse
import os
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch

# Ensure project root is on sys.path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # native-language-identification/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, Config
from src.data.preprocessing import AudioPreprocessor
from src.features.traditional import MFCCExtractor
from src.models.classifier import create_classifier


def load_label_mapping(cfg: Config) -> Dict[str, int]:
    langs = cfg.get('data.languages', []) or []
    return {name: idx for idx, name in enumerate(langs)}


def pick_device(arg_device: str) -> str:
    if arg_device and arg_device.lower() != 'auto':
        return arg_device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True, help='Path to input audio (wav/mp3/flac)')
    ap.add_argument('--config', default='configs/default.yaml', help='Config YAML path')
    ap.add_argument('--checkpoint', default=None, help='Path to trained checkpoint .pt (optional)')
    ap.add_argument('--features', default='mfcc', choices=['mfcc'], help='Feature type (mfcc)')
    ap.add_argument('--arch', default=None, help='Model architecture override (cnn|bilstm|transformer)')
    ap.add_argument('--device', default='auto', help='auto|cpu|cuda')
    ap.add_argument('--topk', type=int, default=3, help='Show top-K predictions')
    args = ap.parse_args()

    # Load config
    cfg_dict = load_config(args.config)
    cfg = Config(cfg_dict)

    # Build label mapping
    label_map = load_label_mapping(cfg)
    if not label_map:
        raise RuntimeError('No languages configured in data.languages')
    idx_to_label = {v: k for k, v in label_map.items()}

    # Device
    device = pick_device(args.device)

    # Preprocessor
    sr = int(cfg.get('data.sample_rate', 16000))
    max_dur = float(cfg.get('data.max_duration', 10))
    min_dur = float(cfg.get('data.min_duration', 0.5))
    pre = AudioPreprocessor(sample_rate=sr, max_duration=max_dur, min_duration=min_dur, normalize=True)

    # Load and preprocess audio
    audio = pre.preprocess(args.audio, trim_silence=True, pad_to_max=True)

    # Feature extraction (MFCC)
    feat_name = args.features.lower()
    if feat_name == 'mfcc':
        mfcc_cfg = cfg.get('features.mfcc', {}) or {}
        fe = MFCCExtractor(
            sample_rate=sr,
            n_mfcc=int(mfcc_cfg.get('n_mfcc', 13)),
            n_fft=int(mfcc_cfg.get('n_fft', 2048)),
            hop_length=int(mfcc_cfg.get('hop_length', 512)),
            n_mels=int(mfcc_cfg.get('n_mels', 128)),
            fmin=float(mfcc_cfg.get('fmin', 0)),
            fmax=float(mfcc_cfg.get('fmax', sr // 2)),
            use_deltas=bool(mfcc_cfg.get('use_deltas', True)),
            use_delta_deltas=bool(mfcc_cfg.get('use_delta_deltas', True)),
            normalize=True
        )
        features = fe(audio)  # (n_features, n_frames)
    else:
        raise NotImplementedError('Only MFCC features are supported in this quick predictor')

    n_features, n_frames = features.shape

    # Model
    arch = (args.arch or cfg.get('model.architecture', 'cnn')).lower()
    model = create_classifier(
        architecture=arch,
        input_dim=n_features,
        num_classes=len(label_map),
        config=cfg_dict.get('model', {})
    ).to(device)

    # Try to load checkpoint
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Default best model location from config
        ckpt_dir = Path(cfg.get('training.checkpoint.save_dir', './models/checkpoints'))
        # try experiment subdir if exists
        exp = cfg.get('experiment_name', 'baseline_experiment')
        candidate = ckpt_dir / exp / 'best_model.pt'
        if candidate.exists():
            checkpoint_path = candidate

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Warning: No trained checkpoint found. Using untrained model — predictions will be unreliable.\n"
              "Train first with ./scripts/train.sh, then re-run with --checkpoint.")

    # Inference
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, F, T)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Top-K
    topk = min(args.topk, len(probs))
    top_idx = np.argsort(probs)[::-1][:topk]
    print("\nPrediction (top-{}):".format(topk))
    for rank, i in enumerate(top_idx, start=1):
        print(f"  {rank}. {idx_to_label[i]} - {probs[i]*100:.2f}%")

    pred_idx = int(top_idx[0])
    print(f"\nDetected language: {idx_to_label[pred_idx]}  (confidence {probs[pred_idx]*100:.2f}%)")


if __name__ == '__main__':
    main()
