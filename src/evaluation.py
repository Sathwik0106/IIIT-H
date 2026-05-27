"""Evaluation helpers shared by project pipelines."""

from __future__ import annotations

import pandas as pd


def accuracy(y_true, y_pred) -> float:
    total = len(y_true)
    correct = sum(str(a) == str(b) for a, b in zip(y_true, y_pred))
    return correct / total if total else 0.0


def confusion_matrix(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for actual, predicted in zip(y_true, y_pred):
        matrix.loc[str(actual), str(predicted)] += 1
    matrix.index.name = "actual"
    matrix.columns.name = "predicted"
    return matrix


def classification_report(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        true_positive = sum(
            str(actual) == label and str(predicted) == label
            for actual, predicted in zip(y_true, y_pred)
        )
        false_positive = sum(
            str(actual) != label and str(predicted) == label
            for actual, predicted in zip(y_true, y_pred)
        )
        false_negative = sum(
            str(actual) == label and str(predicted) != label
            for actual, predicted in zip(y_true, y_pred)
        )
        support = sum(str(actual) == label for actual in y_true)
        precision = true_positive / (true_positive + false_positive or 1)
        recall = true_positive / (true_positive + false_negative or 1)
        f1 = 2 * precision * recall / (precision + recall or 1)
        rows.append(
            {
                "emotion": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )
    return pd.DataFrame(rows)
