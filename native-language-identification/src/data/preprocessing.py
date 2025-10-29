"""
Audio preprocessing module for native language identification.
Handles audio loading, resampling, normalization, and augmentation.
"""

import os
import logging
from typing import Tuple, Optional, Union, List
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from scipy import signal

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocessor for audio data.
    Handles loading, resampling, normalization, and augmentation.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        min_duration: float = 0.5,
        normalize: bool = True
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            normalize: Whether to normalize audio
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize = normalize
        self.max_length = int(sample_rate * max_duration)
        self.min_length = int(sample_rate * min_duration)
    
    def load_audio(
        self,
        audio_path: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            offset: Start reading after this time (in seconds)
            duration: Duration to read (in seconds)
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=offset,
                duration=duration,
                mono=True
            )
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            raise
    
    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Resampled audio array
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        if orig_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def normalize_audio(
        self,
        audio: np.ndarray,
        method: str = "peak"
    ) -> np.ndarray:
        """
        Normalize audio.
        
        Args:
            audio: Audio array
            method: Normalization method ('peak' or 'rms')
            
        Returns:
            Normalized audio array
        """
        if method == "peak":
            # Peak normalization
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio = audio / rms
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return audio
    
    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: int = 20,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim leading and trailing silence.
        
        Args:
            audio: Audio array
            top_db: Threshold in decibels below reference
            frame_length: Frame length for silence detection
            hop_length: Hop length for silence detection
            
        Returns:
            Trimmed audio array
        """
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed
    
    def pad_or_truncate(
        self,
        audio: np.ndarray,
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Pad or truncate audio to target length.
        
        Args:
            audio: Audio array
            target_length: Target length (uses self.max_length if None)
            
        Returns:
            Padded or truncated audio array
        """
        if target_length is None:
            target_length = self.max_length
        
        current_length = len(audio)
        
        if current_length < target_length:
            # Pad with zeros
            pad_length = target_length - current_length
            audio = np.pad(audio, (0, pad_length), mode='constant')
        elif current_length > target_length:
            # Truncate
            audio = audio[:target_length]
        
        return audio
    
    def preprocess(
        self,
        audio_path: Union[str, Path, np.ndarray],
        trim_silence: bool = True,
        pad_to_max: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            audio_path: Path to audio file or audio array
            trim_silence: Whether to trim silence
            pad_to_max: Whether to pad to maximum length
            
        Returns:
            Preprocessed audio array
        """
        # Load audio if path is provided
        if isinstance(audio_path, (str, Path)):
            audio, sr = self.load_audio(audio_path)
        else:
            audio = audio_path
            sr = self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = self.resample(audio, sr, self.sample_rate)
        
        # Trim silence
        if trim_silence:
            audio = self.trim_silence(audio)
        
        # Check minimum length
        if len(audio) < self.min_length:
            logger.warning(f"Audio too short ({len(audio)} < {self.min_length}), padding...")
            audio = self.pad_or_truncate(audio, self.min_length)
        
        # Normalize
        if self.normalize:
            audio = self.normalize_audio(audio)
        
        # Pad or truncate to max length
        if pad_to_max:
            audio = self.pad_or_truncate(audio)
        
        return audio


class AudioAugmentor:
    """
    Audio augmentation for data augmentation during training.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio augmentor.
        
        Args:
            sample_rate: Sample rate of audio
        """
        self.sample_rate = sample_rate
    
    def add_noise(
        self,
        audio: np.ndarray,
        noise_factor: float = 0.005
    ) -> np.ndarray:
        """
        Add Gaussian noise to audio.
        
        Args:
            audio: Audio array
            noise_factor: Standard deviation of noise
            
        Returns:
            Noisy audio array
        """
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented.astype(audio.dtype)
    
    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float = 1.0
    ) -> np.ndarray:
        """
        Time stretch audio.
        
        Args:
            audio: Audio array
            rate: Stretch factor (>1.0 speeds up, <1.0 slows down)
            
        Returns:
            Time-stretched audio array
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: float = 0.0
    ) -> np.ndarray:
        """
        Pitch shift audio.
        
        Args:
            audio: Audio array
            n_steps: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio array
        """
        return librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )
    
    def time_mask(
        self,
        audio: np.ndarray,
        max_mask_size: int = 1000
    ) -> np.ndarray:
        """
        Apply time masking (set random time segment to zero).
        
        Args:
            audio: Audio array
            max_mask_size: Maximum size of mask
            
        Returns:
            Masked audio array
        """
        audio = audio.copy()
        mask_size = np.random.randint(0, max_mask_size)
        mask_start = np.random.randint(0, max(1, len(audio) - mask_size))
        audio[mask_start:mask_start + mask_size] = 0
        return audio
    
    def augment(
        self,
        audio: np.ndarray,
        augmentations: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply random augmentations.
        
        Args:
            audio: Audio array
            augmentations: List of augmentations to apply
            **kwargs: Additional arguments for augmentations
            
        Returns:
            Augmented audio array
        """
        if augmentations is None:
            augmentations = ['noise', 'time_stretch', 'pitch_shift']
        
        augmented = audio.copy()
        
        for aug in augmentations:
            if np.random.rand() > 0.5:  # 50% chance to apply each
                if aug == 'noise':
                    augmented = self.add_noise(augmented, **kwargs.get('noise_kwargs', {}))
                elif aug == 'time_stretch':
                    rate = np.random.uniform(0.9, 1.1)
                    augmented = self.time_stretch(augmented, rate)
                elif aug == 'pitch_shift':
                    n_steps = np.random.uniform(-2, 2)
                    augmented = self.pitch_shift(augmented, n_steps)
                elif aug == 'time_mask':
                    augmented = self.time_mask(augmented, **kwargs.get('mask_kwargs', {}))
        
        return augmented


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        max_duration=10.0,
        normalize=True
    )
    
    # Example: create a dummy audio signal
    dummy_audio = np.random.randn(16000 * 3)  # 3 seconds
    
    # Preprocess
    processed = preprocessor.preprocess(dummy_audio)
    print(f"Processed audio shape: {processed.shape}")
    
    # Create augmentor
    augmentor = AudioAugmentor(sample_rate=16000)
    
    # Augment
    augmented = augmentor.augment(processed)
    print(f"Augmented audio shape: {augmented.shape}")
