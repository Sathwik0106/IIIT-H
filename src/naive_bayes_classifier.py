"""Multinomial Naive Bayes classifier for text emotion recognition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MultinomialNaiveBayes:
    """Small dependency-free multinomial Naive Bayes classifier."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.labels: list[str] = []
        self.class_log_prior: np.ndarray | None = None
        self.feature_log_prob: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.labels = sorted(str(label) for label in np.unique(labels))
        class_feature_counts = []
        class_counts = []
        for label in self.labels:
            rows = features[labels == label]
            class_feature_counts.append(rows.sum(axis=0))
            class_counts.append(rows.shape[0])

        counts = np.vstack(class_feature_counts) + self.alpha
        self.feature_log_prob = np.log(counts / counts.sum(axis=1, keepdims=True))
        class_counts = np.array(class_counts, dtype=np.float32)
        self.class_log_prior = np.log(class_counts / class_counts.sum())

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.class_log_prior is None or self.feature_log_prob is None:
            raise ValueError("Classifier has not been fitted.")
        return features @ self.feature_log_prob.T + self.class_log_prior

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(features)
        return np.array([self.labels[index] for index in np.argmax(scores, axis=1)])

    def save(self, path: str | Path, vocabulary: dict[str, int], config: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "labels": self.labels,
            "vocabulary": vocabulary,
            "config": config,
            "class_log_prior": self.class_log_prior.tolist(),
            "feature_log_prob": self.feature_log_prob.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path
    ) -> tuple["MultinomialNaiveBayes", dict[str, int], dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(alpha=float(payload["alpha"]))
        model.labels = payload["labels"]
        model.class_log_prior = np.array(payload["class_log_prior"], dtype=np.float32)
        model.feature_log_prob = np.array(payload["feature_log_prob"], dtype=np.float32)
        return model, payload["vocabulary"], payload["config"]
