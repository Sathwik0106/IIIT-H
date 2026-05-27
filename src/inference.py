"""Inference utilities for speech, text, and fusion emotion recognition."""

from __future__ import annotations

import json
from pathlib import Path

from src.fusion_features import extract_single_fusion_features
from src.external_text_emotion import ExternalTextEmotionModel
from src.knn_classifier import KNNClassifier
from src.naive_bayes_classifier import MultinomialNaiveBayes
from src.speech_features import extract_speech_features
from src.text_features import TextVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EmotionRecognizer:
    """Load all trained project models and run inference."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = Path(project_root)
        self.speech_model = None
        self.text_model = None
        self.text_vectorizer = None
        self.fusion_model = None
        self.fusion_vectorizer = None
        self.external_text_model = ExternalTextEmotionModel()

    def load_speech(self) -> None:
        self.speech_model, _, _ = KNNClassifier.load(
            self.project_root / "models" / "speech_pipeline" / "speech_model.npz"
        )

    def load_text(self) -> None:
        self.text_model, vocabulary, _ = MultinomialNaiveBayes.load(
            self.project_root / "models" / "text_pipeline" / "text_model.json"
        )
        self.text_vectorizer = TextVectorizer()
        self.text_vectorizer.vocabulary = vocabulary

    def load_fusion(self) -> None:
        self.fusion_model, _, _ = KNNClassifier.load(
            self.project_root / "models" / "fusion_pipeline" / "fusion_model.npz"
        )
        self.fusion_vectorizer = TextVectorizer()
        vocab_path = self.project_root / "models" / "fusion_pipeline" / "fusion_vocabulary.json"
        self.fusion_vectorizer.vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))

    def predict_speech(self, speech_path: str) -> dict:
        if self.speech_model is None:
            self.load_speech()
        features = extract_speech_features(speech_path)[None, :]
        prediction = str(self.speech_model.predict(features)[0])
        return {"model_variant": "speech_only", "emotion": prediction}

    def predict_text(self, text: str) -> dict:
        if self.external_text_model.available():
            return self.external_text_model.predict(text)
        if self.text_model is None or self.text_vectorizer is None:
            self.load_text()
        features = self.text_vectorizer.transform([text])
        prediction = str(self.text_model.predict(features)[0])
        return {
            "model_variant": "text_only_tess_baseline",
            "emotion": prediction,
            "note": "External pretrained text model is not installed; using TESS filename-text baseline.",
        }

    def predict_fusion(self, speech_path: str, text: str) -> dict:
        if self.fusion_model is None or self.fusion_vectorizer is None:
            self.load_fusion()
        features = extract_single_fusion_features(speech_path, text, self.fusion_vectorizer)
        prediction = str(self.fusion_model.predict(features)[0])
        return {"model_variant": "fusion", "emotion": prediction}
