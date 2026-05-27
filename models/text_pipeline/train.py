"""Train the text-only emotion recognition pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import accuracy, classification_report, confusion_matrix
from src.naive_bayes_classifier import MultinomialNaiveBayes
from src.text_features import TextVectorizer


DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "text_pipeline" / "text_model.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the text-only pipeline.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    train_metadata = metadata[metadata["split"] == "train"].reset_index(drop=True)

    vectorizer = TextVectorizer()
    x_train = vectorizer.fit_transform(train_metadata["text"].tolist())
    y_train = train_metadata["emotion"].astype(str).to_numpy()

    classifier = MultinomialNaiveBayes(alpha=1.0)
    classifier.fit(x_train, y_train)
    classifier.save(
        args.model_path,
        vectorizer.vocabulary,
        {
            "pipeline": "text_only",
            "preprocessing": "lowercase_alpha_tokens",
            "features": "word_unigram_and_character_2_3_grams",
            "contextual_model": "bag_of_ngrams_counts",
            "classifier": "multinomial_naive_bayes",
        },
    )

    predictions = classifier.predict(x_train)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_variant": "text_only",
                "split": "train",
                "accuracy": accuracy(y_train, predictions),
                "samples": len(y_train),
            }
        ]
    ).to_csv(args.results_dir / "text_train_accuracy.csv", index=False)
    classification_report(y_train, predictions, classifier.labels).to_csv(
        args.results_dir / "text_train_classification_report.csv", index=False
    )
    confusion_matrix(y_train, predictions, classifier.labels).to_csv(
        args.results_dir / "text_train_confusion_matrix.csv"
    )

    print(f"Saved text model to {args.model_path}")
    print(f"Train accuracy: {accuracy(y_train, predictions):.4f}")


if __name__ == "__main__":
    main()
