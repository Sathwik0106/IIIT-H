"""Train the multimodal speech-text fusion emotion recognition pipeline."""

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
from src.speech_features import extract_speech_features, feature_names
from src.text_features import TextVectorizer


DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_model.npz"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_vocabulary.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"
K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15]
TEXT_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.05]


def build_speech_matrix(metadata: pd.DataFrame) -> np.ndarray:
    return np.vstack([extract_speech_features(path) for path in metadata["speech_path"]])


def validation_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_indices = []
    val_indices = []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        val_count = max(1, round(len(indices) * 0.2))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    return np.array(train_indices), np.array(val_indices)


def tune_fusion(
    fusion_features: np.ndarray,
    labels: np.ndarray,
    speech_feature_count: int,
    text_feature_count: int,
) -> tuple[int, float, pd.DataFrame]:
    train_indices, val_indices = validation_split(labels)
    rows = []
    best_k = K_CANDIDATES[0]
    best_weight = TEXT_WEIGHTS[0]
    best_score = -1.0
    for text_weight in TEXT_WEIGHTS:
        feature_weights = np.concatenate(
            [
                np.ones(speech_feature_count, dtype=np.float32),
                np.full(text_feature_count, text_weight, dtype=np.float32),
            ]
        )
        for k in K_CANDIDATES:
            classifier = KNNClassifier(
                k=k, feature_weights=feature_weights, distance_weighted=True
            )
            classifier.fit(fusion_features[train_indices], labels[train_indices])
            predictions = classifier.predict(fusion_features[val_indices])
            score = accuracy(labels[val_indices], predictions)
            rows.append(
                {
                    "k": k,
                    "text_branch_weight": text_weight,
                    "validation_accuracy": score,
                }
            )
            if score > best_score:
                best_k = k
                best_weight = text_weight
                best_score = score
    return best_k, best_weight, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fusion pipeline.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    train_metadata = metadata[metadata["split"] == "train"].reset_index(drop=True)

    text_vectorizer = TextVectorizer()
    text_features = text_vectorizer.fit_transform(train_metadata["text"].tolist())
    speech_features = build_speech_matrix(train_metadata)
    fusion_features = combine_features(speech_features, text_features)
    y_train = train_metadata["emotion"].astype(str).to_numpy()

    best_k, best_text_weight, tuning_table = tune_fusion(
        fusion_features,
        y_train,
        speech_features.shape[1],
        text_features.shape[1],
    )

    feature_weights = np.concatenate(
        [
            np.ones(speech_features.shape[1], dtype=np.float32),
            np.full(text_features.shape[1], best_text_weight, dtype=np.float32),
        ]
    )
    classifier = KNNClassifier(
        k=best_k, feature_weights=feature_weights, distance_weighted=True
    )
    classifier.fit(fusion_features, y_train)
    text_feature_names = [
        term for term, _ in sorted(text_vectorizer.vocabulary.items(), key=lambda item: item[1])
    ]
    classifier.save(
        args.model_path,
        feature_names() + [f"text_{name}" for name in text_feature_names],
        {
            "pipeline": "fusion",
            "fusion": "early_concatenation",
            "text_branch_weight": best_text_weight,
            "speech_branch": "temporal_acoustic_statistics",
            "text_branch": "word_and_character_ngram_counts",
            "classifier": "knn",
            "k": best_k,
            "distance_weighted": True,
        },
    )
    args.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_path.write_text(
        json.dumps(text_vectorizer.vocabulary, indent=2), encoding="utf-8"
    )

    predictions = classifier.predict(fusion_features)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_variant": "fusion",
                "split": "train",
                "accuracy": accuracy(y_train, predictions),
                "samples": len(y_train),
            }
        ]
    ).to_csv(args.results_dir / "fusion_train_accuracy.csv", index=False)
    classification_report(y_train, predictions, classifier.labels).to_csv(
        args.results_dir / "fusion_train_classification_report.csv", index=False
    )
    confusion_matrix(y_train, predictions, classifier.labels).to_csv(
        args.results_dir / "fusion_train_confusion_matrix.csv"
    )
    tuning_table.to_csv(args.results_dir / "fusion_knn_tuning.csv", index=False)

    print(f"Saved fusion model to {args.model_path}")
    print(f"Selected k: {best_k}")
    print(f"Selected text branch weight: {best_text_weight}")
    print(f"Train accuracy: {accuracy(y_train, predictions):.4f}")


if __name__ == "__main__":
    main()
