#!/usr/bin/env python3
"""
Ingest an uploaded audio file into the project and run prediction.

This will:
  - Copy the provided audio into data/uploads/<timestamp>_<basename>
  - Run the predictor (MFCC-based) to get top-K language predictions
  - Save a JSON with results at outputs/predictions/<same_basename>.json
  - Print a compact summary to stdout

Usage:
  python3 scripts/ingest_and_predict.py \
    --audio /path/to/your.wav \
    [--config configs/default.yaml] \
    [--checkpoint models/checkpoints/<exp>/best_model.pt] \
    [--topk 5]
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

# Local imports
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, Config
from src.data.preprocessing import AudioPreprocessor
from src.features.traditional import MFCCExtractor
from src.models.classifier import create_classifier


def pick_device(arg_device: str) -> str:
    if arg_device and arg_device.lower() != 'auto':
        return arg_device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True, help='Path to uploaded audio file (wav/mp3/flac)')
    ap.add_argument('--config', default='configs/default.yaml', help='Config YAML path')
    ap.add_argument('--checkpoint', default=None, help='Path to trained checkpoint .pt (optional)')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    # Resolve paths
    audio_src = Path(args.audio).expanduser().resolve()
    if not audio_src.exists():
        raise FileNotFoundError(f"Audio not found: {audio_src}")

    uploads_dir = PROJECT_ROOT / 'data' / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_name = f"{timestamp}_{audio_src.name}"
    audio_dst = uploads_dir / dest_name
    shutil.copy2(audio_src, audio_dst)

    # Load config
    cfg_dict = load_config(args.config)
    cfg = Config(cfg_dict)
    languages = cfg.get('data.languages', []) or []
    if not languages:
        raise RuntimeError('No languages configured in data.languages')
    label_map = {name: idx for idx, name in enumerate(languages)}
    idx_to_label = {v: k for k, v in label_map.items()}

    # Device and preprocessing
    device = pick_device(args.device)
    sr = int(cfg.get('data.sample_rate', 16000))
    pre = AudioPreprocessor(sample_rate=sr,
                            max_duration=float(cfg.get('data.max_duration', 10)),
                            min_duration=float(cfg.get('data.min_duration', 0.5)),
                            normalize=True)
    audio = pre.preprocess(audio_dst.as_posix(), trim_silence=True, pad_to_max=True)

    # Features (MFCC)
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
    features = fe(audio)  # (F, T)

    # Model
    arch = (cfg.get('model.architecture', 'cnn') or 'cnn').lower()
    model = create_classifier(architecture=arch,
                              input_dim=features.shape[0],
                              num_classes=len(languages),
                              config=cfg_dict.get('model', {})).to(device)

    # Load checkpoint if provided or discover default
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dir = Path(cfg.get('training.checkpoint.save_dir', './models/checkpoints'))
        exp = cfg.get('experiment_name', 'baseline_experiment')
        candidate = (PROJECT_ROOT / ckpt_dir / exp / 'best_model.pt').resolve()
        if candidate.exists():
            ckpt_path = candidate
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        loaded = True
    else:
        loaded = False

    # Predict
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Top-K
    topk = min(args.topk, len(probs))
    order = np.argsort(probs)[::-1]
    top_idx = order[:topk]
    top = [{
        'label': idx_to_label[int(i)],
        'probability': float(probs[int(i)])
    } for i in top_idx]

    result = {
        'audio_uploaded': audio_dst.as_posix(),
        'sample_rate': sr,
        'model_architecture': arch,
        'checkpoint_loaded': loaded,
        'detected_language': idx_to_label[int(top_idx[0])],
        'confidence': float(probs[int(top_idx[0])]),
        'topk': top,
        'all_labels': languages,
    }

    # Save JSON
    out_dir = PROJECT_ROOT / 'outputs' / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / (audio_dst.stem + '.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    # Print compact summary
    print("Uploaded to:", audio_dst)
    if loaded:
        print("Checkpoint:", ckpt_path)
    else:
        print("Warning: No trained checkpoint loaded (demo mode)")
    print("Top-{}:".format(topk))
    for i, item in enumerate(top, 1):
        print(f"  {i}. {item['label']} - {item['probability']*100:.2f}%")
    print("Saved prediction JSON:", out_json)


if __name__ == '__main__':
    main()
