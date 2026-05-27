"""Evaluate the multimodal speech-text fusion emotion recognition pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import accuracy, classification_report, confusion_matrix
from src.fusion_features import combine_features
from src.knn_classifier import KNNClassifier
from src.speech_features import extract_speech_features
from src.text_features import TextVectorizer


DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_model.npz"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_vocabulary.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"


def build_speech_matrix(metadata: pd.DataFrame) -> np.ndarray:
    return np.vstack([extract_speech_features(path) for path in metadata["speech_path"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fusion pipeline.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    classifier, _, _ = KNNClassifier.load(args.model_path)
    vectorizer = TextVectorizer()
    vectorizer.vocabulary = json.loads(args.vocab_path.read_text(encoding="utf-8"))

    metadata = pd.read_csv(args.metadata_path)
    test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    speech_features = build_speech_matrix(test_metadata)
    text_features = vectorizer.transform(test_metadata["text"].tolist())
    x_test = combine_features(speech_features, text_features)
    y_test = test_metadata["emotion"].astype(str).to_numpy()
    predictions = classifier.predict(x_test)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_variant": "fusion",
                "split": "test",
                "accuracy": accuracy(y_test, predictions),
                "samples": len(y_test),
            }
        ]
    ).to_csv(args.results_dir / "fusion_test_accuracy.csv", index=False)
    classification_report(y_test, predictions, classifier.labels).to_csv(
        args.results_dir / "fusion_test_classification_report.csv", index=False
    )
    confusion_matrix(y_test, predictions, classifier.labels).to_csv(
        args.results_dir / "fusion_test_confusion_matrix.csv"
    )
    test_metadata.assign(predicted_emotion=predictions).to_csv(
        args.results_dir / "fusion_test_predictions.csv", index=False
    )

    print(f"Test accuracy: {accuracy(y_test, predictions):.4f}")


if __name__ == "__main__":
    main()
