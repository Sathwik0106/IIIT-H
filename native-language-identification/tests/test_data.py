"""Tests for data loading module."""

import pytest
import numpy as np
from src.data import AudioPreprocessor, AudioAugmentor


def test_audio_preprocessor():
    """Test audio preprocessor."""
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        max_duration=3.0,
        normalize=True
    )
    
    # Create dummy audio
    audio = np.random.randn(16000 * 2)  # 2 seconds
    
    # Preprocess
    processed = preprocessor.preprocess(audio, pad_to_max=True)
    
    # Check output shape
    expected_length = 16000 * 3  # 3 seconds
    assert processed.shape[0] == expected_length
    
    # Check normalization
    assert np.abs(np.abs(processed).max() - 1.0) < 0.1 or np.abs(processed).max() < 1.0


def test_audio_augmentor():
    """Test audio augmentation."""
    augmentor = AudioAugmentor(sample_rate=16000)
    
    # Create dummy audio
    audio = np.random.randn(16000)
    
    # Test noise augmentation
    noisy = augmentor.add_noise(audio, noise_factor=0.01)
    assert noisy.shape == audio.shape
    assert not np.array_equal(noisy, audio)
    
    # Test time stretch
    stretched = augmentor.time_stretch(audio, rate=1.1)
    assert stretched.shape[0] != audio.shape[0]
    
    # Test time mask
    masked = augmentor.time_mask(audio, max_mask_size=100)
    assert masked.shape == audio.shape


if __name__ == "__main__":
    pytest.main([__file__])
