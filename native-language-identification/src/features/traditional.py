"""
Traditional acoustic feature extraction (MFCCs).
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np
import librosa
import torch

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """
    Extracts MFCC features from audio signals.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0,
        fmax: Optional[float] = None,
        use_deltas: bool = True,
        use_delta_deltas: bool = True,
        normalize: bool = True
    ):
        """
        Initialize MFCC extractor.
        
        Args:
            sample_rate: Sample rate of audio
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sr/2)
            use_deltas: Whether to compute delta features
            use_delta_deltas: Whether to compute delta-delta features
            normalize: Whether to normalize features
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate // 2
        self.use_deltas = use_deltas
        self.use_delta_deltas = use_delta_deltas
        self.normalize = normalize
        
        logger.info(f"Initialized MFCC extractor with n_mfcc={n_mfcc}")
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal array
            
        Returns:
            MFCC features array of shape (n_features, n_frames)
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        features = [mfccs]
        
        # Compute deltas (first-order derivatives)
        if self.use_deltas:
            deltas = librosa.feature.delta(mfccs)
            features.append(deltas)
        
        # Compute delta-deltas (second-order derivatives)
        if self.use_delta_deltas:
            delta_deltas = librosa.feature.delta(mfccs, order=2)
            features.append(delta_deltas)
        
        # Concatenate all features
        features = np.concatenate(features, axis=0)
        
        # Normalize if requested
        if self.normalize:
            features = self._normalize(features)
        
        return features
    
    def extract_statistics(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract statistical features (mean, std) from MFCCs.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Statistical features array
        """
        mfccs = self.extract(audio)
        
        # Compute statistics across time
        mean = np.mean(mfccs, axis=1)
        std = np.std(mfccs, axis=1)
        
        # Concatenate mean and std
        stats = np.concatenate([mean, std])
        
        return stats
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Feature array
            
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (features - mean) / std
        return normalized
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Callable interface for feature extraction.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Extracted features
        """
        return self.extract(audio)


class SpectralFeatureExtractor:
    """
    Extracts additional spectral features.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize spectral feature extractor.
        
        Args:
            sample_rate: Sample rate of audio
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid."""
        return librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral rolloff."""
        return librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral contrast."""
        return librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate."""
        return librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
    
    def extract_all(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all spectral features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Dictionary of spectral features
        """
        return {
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'spectral_rolloff': self.extract_spectral_rolloff(audio),
            'spectral_contrast': self.extract_spectral_contrast(audio),
            'zero_crossing_rate': self.extract_zero_crossing_rate(audio)
        }


class CombinedFeatureExtractor:
    """
    Combines MFCC and spectral features.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mfcc_params: Optional[Dict] = None,
        include_spectral: bool = True
    ):
        """
        Initialize combined feature extractor.
        
        Args:
            sample_rate: Sample rate of audio
            mfcc_params: Parameters for MFCC extraction
            include_spectral: Whether to include spectral features
        """
        self.sample_rate = sample_rate
        
        # Initialize MFCC extractor
        mfcc_params = mfcc_params or {}
        self.mfcc_extractor = MFCCExtractor(
            sample_rate=sample_rate,
            **mfcc_params
        )
        
        # Initialize spectral feature extractor
        self.include_spectral = include_spectral
        if include_spectral:
            self.spectral_extractor = SpectralFeatureExtractor(
                sample_rate=sample_rate
            )
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract combined features.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Combined features array
        """
        # Extract MFCCs
        mfcc_features = self.mfcc_extractor.extract(audio)
        
        if not self.include_spectral:
            return mfcc_features
        
        # Extract spectral features
        spectral_features = self.spectral_extractor.extract_all(audio)
        
        # Combine features
        # Ensure all features have the same number of frames
        n_frames = mfcc_features.shape[1]
        
        combined = [mfcc_features]
        
        for name, feature in spectral_features.items():
            if feature.ndim == 1:
                feature = feature.reshape(1, -1)
            
            # Resize to match MFCC frames if needed
            if feature.shape[1] != n_frames:
                # Simple interpolation
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, feature.shape[1])
                x_new = np.linspace(0, 1, n_frames)
                f = interp1d(x_old, feature, axis=1, kind='linear')
                feature = f(x_new)
            
            combined.append(feature)
        
        # Concatenate all features
        combined_features = np.concatenate(combined, axis=0)
        
        return combined_features
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Callable interface for feature extraction.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Extracted features
        """
        return self.extract(audio)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy audio
    sample_rate = 16000
    duration = 3  # seconds
    audio = np.random.randn(sample_rate * duration)
    
    # Extract MFCCs
    mfcc_extractor = MFCCExtractor(
        sample_rate=sample_rate,
        n_mfcc=13,
        use_deltas=True,
        use_delta_deltas=True
    )
    
    mfccs = mfcc_extractor.extract(audio)
    print(f"MFCC features shape: {mfccs.shape}")
    
    # Extract statistics
    stats = mfcc_extractor.extract_statistics(audio)
    print(f"Statistical features shape: {stats.shape}")
    
    # Extract combined features
    combined_extractor = CombinedFeatureExtractor(
        sample_rate=sample_rate,
        include_spectral=True
    )
    
    combined = combined_extractor.extract(audio)
    print(f"Combined features shape: {combined.shape}")
