"""
Audio utility functions.
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    info = sf.info(audio_path)
    return info.duration


def get_audio_info(audio_path: str) -> dict:
    """
    Get information about audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    info = sf.info(audio_path)
    return {
        'duration': info.duration,
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'format': info.format,
        'subtype': info.subtype
    }


def save_audio(
    audio: np.ndarray,
    save_path: str,
    sample_rate: int = 16000
):
    """
    Save audio array to file.
    
    Args:
        audio: Audio array
        save_path: Path to save audio
        sample_rate: Sample rate
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(save_path, audio, sample_rate)
    logger.info(f"Saved audio to {save_path}")


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Waveform",
    save_path: Optional[str] = None
):
    """
    Plot audio waveform.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        title: Plot title
        save_path: Path to save plot (if None, displays plot)
    """
    plt.figure(figsize=(12, 4))
    time = np.arange(len(audio)) / sample_rate
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved waveform plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Spectrogram",
    save_path: Optional[str] = None
):
    """
    Plot spectrogram of audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 4))
    
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    librosa.display.specshow(
        D,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spectrogram plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mfcc(
    mfcc: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 512,
    title: str = "MFCC",
    save_path: Optional[str] = None
):
    """
    Plot MFCC features.
    
    Args:
        mfcc: MFCC features array
        sample_rate: Sample rate
        hop_length: Hop length
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 4))
    
    librosa.display.specshow(
        mfcc,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time'
    )
    
    plt.colorbar()
    plt.ylabel('MFCC Coefficients')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved MFCC plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy audio
    sample_rate = 16000
    duration = 3
    audio = np.random.randn(sample_rate * duration) * 0.1
    
    # Plot waveform
    plot_waveform(audio, sample_rate)
