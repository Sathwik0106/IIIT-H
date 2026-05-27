"""Optional external pretrained text-emotion model integration."""

from __future__ import annotations


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "pleasant_surprise",
}


class ExternalTextEmotionModel:
    """Lazy wrapper around a Hugging Face pretrained emotion classifier."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._classifier = None

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def load(self) -> None:
        if self._classifier is not None:
            return
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "External text model requires transformers and torch. "
                "Install requirements.txt to enable it."
            ) from exc
        self._classifier = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
        )

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Text is required for external text prediction.")
        self.load()
        scores = self._classifier(text)[0]
        best = max(scores, key=lambda item: item["score"])
        raw_label = best["label"].lower()
        return {
            "model_variant": "text_external_pretrained",
            "emotion": LABEL_MAP.get(raw_label, raw_label),
            "raw_label": raw_label,
            "confidence": float(best["score"]),
            "model_name": self.model_name,
        }
