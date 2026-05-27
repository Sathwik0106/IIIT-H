"""Small k-nearest-neighbors classifier for speech emotion recognition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EPSILON = 1e-6


class KNNClassifier:
    """Euclidean kNN classifier with saved standardization statistics."""

    def __init__(
        self,
        k: int = 7,
        feature_weights: np.ndarray | None = None,
        distance_weighted: bool = True,
    ) -> None:
        self.k = k
        self.feature_weights = feature_weights
        self.distance_weighted = distance_weighted
        self.labels: list[str] = []
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.train_features: np.ndarray | None = None
        self.train_labels: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.labels = sorted(str(label) for label in np.unique(labels))
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0) + EPSILON
        self.train_features = self._scale(features)
        self.train_labels = labels.astype(str)

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = self._scale(features)
        predictions = []
        for row in scaled:
            distances = np.sum((self.train_features - row) ** 2, axis=1)
            neighbors = np.argpartition(distances, self.k - 1)[: self.k]
            votes = {label: 0 for label in self.labels}
            distance_sums = {label: 0.0 for label in self.labels}
            for index in neighbors:
                label = str(self.train_labels[index])
                if self.distance_weighted:
                    votes[label] += 1.0 / (float(distances[index]) + EPSILON)
                else:
                    votes[label] += 1
                distance_sums[label] += float(distances[index])
            predictions.append(
                max(
                    self.labels,
                    key=lambda label: (
                        votes[label],
                        -distance_sums[label],
                        -self.labels.index(label),
                    ),
                )
            )
        return np.array(predictions)

    def save(self, path: str | Path, feature_names: list[str], config: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "k": self.k,
            "labels": self.labels,
            "feature_names": feature_names,
            "config": config,
            "has_feature_weights": self.feature_weights is not None,
            "distance_weighted": self.distance_weighted,
        }
        feature_weights = (
            self.feature_weights
            if self.feature_weights is not None
            else np.ones_like(self.scaler_mean)
        )
        np.savez_compressed(
            path,
            metadata=json.dumps(metadata),
            scaler_mean=self.scaler_mean,
            scaler_std=self.scaler_std,
            feature_weights=feature_weights,
            train_features=self.train_features,
            train_labels=self.train_labels,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["KNNClassifier", list[str], dict]:
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        feature_weights = (
            payload["feature_weights"] if metadata.get("has_feature_weights") else None
        )
        model = cls(
            k=int(metadata["k"]),
            feature_weights=feature_weights,
            distance_weighted=bool(metadata.get("distance_weighted", True)),
        )
        model.labels = metadata["labels"]
        model.scaler_mean = payload["scaler_mean"]
        model.scaler_std = payload["scaler_std"]
        model.train_features = payload["train_features"]
        model.train_labels = payload["train_labels"]
        return model, metadata["feature_names"], metadata["config"]

    def _scale(self, features: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Classifier has not been fitted.")
        scaled = (features - self.scaler_mean) / self.scaler_std
        if self.feature_weights is not None:
            scaled = scaled * self.feature_weights
        return scaled
