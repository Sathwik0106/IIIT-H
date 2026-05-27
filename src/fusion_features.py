"""Feature utilities for multimodal speech-text fusion."""

from __future__ import annotations

import numpy as np

from src.speech_features import extract_speech_features


def combine_features(speech_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Concatenate speech and text representations for early fusion."""
    return np.hstack([speech_features, text_features]).astype(np.float32)


def extract_single_fusion_features(speech_path: str, text: str, vectorizer) -> np.ndarray:
    """Extract a single early-fusion representation."""
    speech = extract_speech_features(speech_path)[None, :]
    text_vector = vectorizer.transform([text])
    return combine_features(speech, text_vector)
