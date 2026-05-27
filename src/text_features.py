"""Text preprocessing and feature extraction for the text-only pipeline."""

from __future__ import annotations

import re

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z]+")


def clean_text(text: str) -> str:
    """Lowercase text and keep alphabetic tokens."""
    tokens = TOKEN_PATTERN.findall(str(text).lower())
    return " ".join(tokens)


def extract_ngrams(text: str) -> list[str]:
    """Extract word and character cues from the cleaned transcript text."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    features = [f"word={token}" for token in tokens]
    compact = cleaned.replace(" ", "")
    for ngram_size in (2, 3):
        for index in range(max(0, len(compact) - ngram_size + 1)):
            features.append(f"char{ngram_size}={compact[index:index + ngram_size]}")
    return features


class TextVectorizer:
    """Count vectorizer for the short TESS filename transcripts."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def fit(self, texts) -> None:
        terms = sorted({term for text in texts for term in extract_ngrams(text)})
        self.vocabulary = {term: index for index, term in enumerate(terms)}

    def transform(self, texts) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for term in extract_ngrams(text):
                column_index = self.vocabulary.get(term)
                if column_index is not None:
                    matrix[row_index, column_index] += 1.0
        return matrix

    def fit_transform(self, texts) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
