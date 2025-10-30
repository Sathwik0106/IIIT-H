#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../native-language-identification
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, Config
from src.data.preprocessing import AudioPreprocessor
from src.features.traditional import MFCCExtractor
from src.models.classifier import create_classifier


def build_app():
    app = Flask(__name__)
    CORS(app)

    cfg_path = PROJECT_ROOT / 'configs' / 'default.yaml'
    cfg = Config(load_config(cfg_path.as_posix()))

    languages = cfg.get('data.languages', []) or []
    label_map = {name: idx for idx, name in enumerate(languages)}
    idx_to_label = {v: k for k, v in label_map.items()}
    cuisine_map = cfg.get('application.cuisine_mapping', {}) or {}

    # Preprocessor and feature extractor
    sr = int(cfg.get('data.sample_rate', 16000))
    pre = AudioPreprocessor(sample_rate=sr,
                            max_duration=float(cfg.get('data.max_duration', 10)),
                            min_duration=float(cfg.get('data.min_duration', 0.5)),
                            normalize=True)
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

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = (cfg.get('model.architecture', 'cnn') or 'cnn').lower()
    dummy_features = fe(np.zeros(sr, dtype=np.float32))  # (F, T)
    model = create_classifier(architecture=arch,
                              input_dim=dummy_features.shape[0],
                              num_classes=len(languages),
                              config=cfg.to_dict().get('model', {})).to(device)

    # Try to load checkpoint
    ckpt_dir = Path(cfg.get('training.checkpoint.save_dir', './models/checkpoints'))
    exp = cfg.get('experiment_name', 'baseline_experiment')
    candidate = (PROJECT_ROOT / ckpt_dir / exp / 'best_model.pt').resolve()
    checkpoint_loaded = False
    if candidate.exists():
        ckpt = torch.load(candidate, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        checkpoint_loaded = True

    uploads_dir = PROJECT_ROOT / 'data' / 'uploads'
    outputs_dir = PROJECT_ROOT / 'outputs' / 'predictions'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    @app.get('/api/health')
    def health():
        return jsonify({
            'status': 'ok',
            'languages': languages,
            'checkpoint_loaded': checkpoint_loaded,
            'device': device
        })

    @app.post('/api/predict')
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided under field "file"'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Save upload into project
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = f.filename.replace(' ', '_')
        dest = uploads_dir / f"{ts}_{safe_name}"
        f.save(dest.as_posix())

        # Preprocess and features
        audio = pre.preprocess(dest.as_posix(), trim_silence=True, pad_to_max=True)
        feats = fe(audio)  # (F, T)

        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        order = np.argsort(probs)[::-1]
        topk = int(request.form.get('topk', 5))
        topk = min(topk, len(probs))
        top = [{
            'label': idx_to_label[int(i)],
            'probability': float(probs[int(i)])
        } for i in order[:topk]]

        detected_idx = int(order[0])
        detected_label = idx_to_label[detected_idx]
        confidence = float(probs[detected_idx])
        cuisines = cuisine_map.get(detected_label, [])

        result = {
            'uploaded_path': dest.as_posix(),
            'detected_language': detected_label,
            'confidence': confidence,
            'checkpoint_loaded': checkpoint_loaded,
            'top': top,
            'cuisine_recommendations': cuisines,
            'all_labels': languages
        }
        return jsonify(result)

    return app


if __name__ == '__main__':
    app = build_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
