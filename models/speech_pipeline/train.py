"""Train the speech-only emotion recognition pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import accuracy, classification_report, confusion_matrix
from src.knn_classifier import KNNClassifier
from src.speech_features import (
    FRAME_MS,
    HOP_MS,
    SAMPLE_RATE,
    extract_speech_features,
    feature_names,
)


DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "speech_pipeline" / "speech_model.npz"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"
K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15]


def build_feature_matrix(metadata: pd.DataFrame) -> np.ndarray:
    features = []
    for row in metadata.itertuples(index=False):
        features.append(extract_speech_features(row.speech_path))
    return np.vstack(features)


def validation_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_indices = []
    val_indices = []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        val_count = max(1, round(len(indices) * 0.2))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    return np.array(train_indices), np.array(val_indices)


def choose_best_k(features: np.ndarray, labels: np.ndarray) -> tuple[int, pd.DataFrame]:
    train_indices, val_indices = validation_split(labels)
    rows = []
    best_k = K_CANDIDATES[0]
    best_score = -1.0
    for k in K_CANDIDATES:
        classifier = KNNClassifier(k=k, distance_weighted=True)
        classifier.fit(features[train_indices], labels[train_indices])
        predictions = classifier.predict(features[val_indices])
        score = accuracy(labels[val_indices], predictions)
        rows.append({"k": k, "validation_accuracy": score})
        if score > best_score:
            best_k = k
            best_score = score
    return best_k, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the speech-only pipeline.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    train_metadata = metadata[metadata["split"] == "train"].reset_index(drop=True)

    x_train = build_feature_matrix(train_metadata)
    y_train = train_metadata["emotion"].astype(str).to_numpy()

    best_k, tuning_table = choose_best_k(x_train, y_train)

    classifier = KNNClassifier(k=best_k, distance_weighted=True)
    classifier.fit(x_train, y_train)
    classifier.save(
        args.model_path,
        feature_names(),
        {
            "pipeline": "speech_only",
            "sample_rate": SAMPLE_RATE,
            "frame_ms": FRAME_MS,
            "hop_ms": HOP_MS,
            "classifier": "knn",
            "k": best_k,
            "distance_weighted": True,
            "feature_set": "spectral_energy_mfcc_temporal_statistics",
        },
    )

    train_predictions = classifier.predict(x_train)
    labels = classifier.labels
    args.results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "model_variant": "speech_only",
                "split": "train",
                "accuracy": accuracy(y_train, train_predictions),
                "samples": len(y_train),
            }
        ]
    ).to_csv(args.results_dir / "speech_train_accuracy.csv", index=False)

    classification_report(y_train, train_predictions, labels).to_csv(
        args.results_dir / "speech_train_classification_report.csv", index=False
    )
    confusion_matrix(y_train, train_predictions, labels).to_csv(
        args.results_dir / "speech_train_confusion_matrix.csv"
    )
    tuning_table.to_csv(args.results_dir / "speech_knn_tuning.csv", index=False)

    print(f"Saved speech model to {args.model_path}")
    print(f"Selected k: {best_k}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.4f}")


if __name__ == "__main__":
    main()
