"""Test package initialization."""

import pytest


def test_imports():
    """Test that main packages can be imported."""
    try:
        from src.data import IndicAccentDataLoader
        from src.features import MFCCExtractor, HuBERTFeatureExtractor
        from src.models import create_classifier
        from src.training import Trainer
        from src.evaluation import Evaluator
        from src.utils import load_config
        from src.application import CuisineRecommender
        
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
