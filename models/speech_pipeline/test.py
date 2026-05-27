"""Evaluate the speech-only emotion recognition pipeline."""

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
from src.speech_features import extract_speech_features


DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "speech_pipeline" / "speech_model.npz"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"


def build_feature_matrix(metadata: pd.DataFrame) -> np.ndarray:
    features = []
    for row in metadata.itertuples(index=False):
        features.append(extract_speech_features(row.speech_path))
    return np.vstack(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the speech-only pipeline.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    classifier, _, _ = KNNClassifier.load(args.model_path)
    metadata = pd.read_csv(args.metadata_path)
    test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

    x_test = build_feature_matrix(test_metadata)
    y_test = test_metadata["emotion"].astype(str).to_numpy()
    predictions = classifier.predict(x_test)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_variant": "speech_only",
                "split": "test",
                "accuracy": accuracy(y_test, predictions),
                "samples": len(y_test),
            }
        ]
    ).to_csv(args.results_dir / "speech_test_accuracy.csv", index=False)

    classification_report(y_test, predictions, classifier.labels).to_csv(
        args.results_dir / "speech_test_classification_report.csv", index=False
    )
    confusion_matrix(y_test, predictions, classifier.labels).to_csv(
        args.results_dir / "speech_test_confusion_matrix.csv"
    )
    test_metadata.assign(predicted_emotion=predictions).to_csv(
        args.results_dir / "speech_test_predictions.csv", index=False
    )

    print(f"Test accuracy: {accuracy(y_test, predictions):.4f}")


if __name__ == "__main__":
    main()
