"""Speech preprocessing and feature extraction for the speech-only pipeline."""

from __future__ import annotations

import wave
import aifc
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
FRAME_MS = 25
HOP_MS = 10
EPSILON = 1e-8


BASE_FEATURE_NAMES = [
    "rms",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
]
MFCC_COUNT = 13
MEL_BANDS = 26


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit PCM wav file as float32 audio."""
    path = Path(path)
    header = path.read_bytes()[:4]
    if header == b"FORM":
        with aifc.open(str(path), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            frames = audio_file.readframes(audio_file.getnframes())
        dtype = ">i2"
    else:
        with wave.open(str(path), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            frames = audio_file.readframes(audio_file.getnframes())
        dtype = "<i2"

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM wav, got sample width {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio with deterministic linear interpolation."""
    if source_rate == target_rate:
        return audio
    if audio.size == 0:
        return audio

    duration = audio.size / source_rate
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0, audio.size - 1, num=audio.size)
    target_positions = np.linspace(0, audio.size - 1, num=target_size)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold_ratio: float = 0.02) -> np.ndarray:
    """Trim leading and trailing low-amplitude regions."""
    if audio.size == 0:
        return audio

    threshold = max(float(np.max(np.abs(audio))) * threshold_ratio, EPSILON)
    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio
    return audio[active[0] : active[-1] + 1]


def frame_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert audio into overlapping frames."""
    frame_length = int(sample_rate * FRAME_MS / 1000)
    hop_length = int(sample_rate * HOP_MS / 1000)
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    frame_count = 1 + int(np.floor((audio.size - frame_length) / hop_length))
    frames = np.empty((frame_count, frame_length), dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_length
        frames[index] = audio[start : start + frame_length]
    return frames * np.hanning(frame_length).astype(np.float32)


def extract_frame_features(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract frame-level acoustic cues."""
    rms = np.sqrt(np.mean(frames**2, axis=1) + EPSILON)
    zero_crossings = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    spectrum = np.abs(np.fft.rfft(frames, axis=1)) + EPSILON
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    spectrum_sum = np.sum(spectrum, axis=1) + EPSILON

    centroid = np.sum(spectrum * freqs, axis=1) / spectrum_sum
    bandwidth = np.sqrt(
        np.sum(spectrum * (freqs[None, :] - centroid[:, None]) ** 2, axis=1)
        / spectrum_sum
    )

    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_threshold = 0.85 * spectrum_sum
    rolloff_bins = np.argmax(cumulative >= rolloff_threshold[:, None], axis=1)
    rolloff = freqs[rolloff_bins]

    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)

    mfcc = extract_mfcc(spectrum, sample_rate, frames.shape[1])

    return np.column_stack(
        [rms, zero_crossings, centroid, bandwidth, rolloff, flatness, mfcc]
    ).astype(np.float32)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, fft_size: int) -> np.ndarray:
    """Build a triangular mel filterbank."""
    low_mel = hz_to_mel(np.array([0.0]))[0]
    high_mel = hz_to_mel(np.array([sample_rate / 2.0]))[0]
    mel_points = np.linspace(low_mel, high_mel, MEL_BANDS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    bank = np.zeros((MEL_BANDS, fft_size // 2 + 1), dtype=np.float32)
    for band in range(1, MEL_BANDS + 1):
        left, center, right = bins[band - 1], bins[band], bins[band + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for index in range(left, min(center, bank.shape[1])):
            bank[band - 1, index] = (index - left) / (center - left)
        for index in range(center, min(right, bank.shape[1])):
            bank[band - 1, index] = (right - index) / (right - center)
    return bank


def dct_basis(input_size: int, output_size: int) -> np.ndarray:
    basis = np.empty((output_size, input_size), dtype=np.float32)
    scale = np.sqrt(2.0 / input_size)
    for row in range(output_size):
        for column in range(input_size):
            basis[row, column] = scale * np.cos(
                np.pi * row * (2 * column + 1) / (2 * input_size)
            )
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def extract_mfcc(spectrum: np.ndarray, sample_rate: int, frame_length: int) -> np.ndarray:
    """Compute compact MFCC-style features from the magnitude spectrum."""
    power = spectrum**2
    filters = mel_filterbank(sample_rate, frame_length)
    mel_energy = np.maximum(power @ filters.T, EPSILON)
    log_mel = np.log(mel_energy)
    return log_mel @ dct_basis(MEL_BANDS, MFCC_COUNT).T


def temporal_summary(frame_features: np.ndarray) -> np.ndarray:
    """Summarize frame-level cues into one utterance representation."""
    means = np.mean(frame_features, axis=0)
    stds = np.std(frame_features, axis=0)
    mins = np.min(frame_features, axis=0)
    maxs = np.max(frame_features, axis=0)

    if frame_features.shape[0] > 1:
        deltas = np.diff(frame_features, axis=0)
        delta_means = np.mean(np.abs(deltas), axis=0)
    else:
        delta_means = np.zeros(frame_features.shape[1], dtype=np.float32)

    return np.concatenate([means, stds, mins, maxs, delta_means]).astype(np.float32)


def feature_names() -> list[str]:
    """Return names for the utterance-level speech representation."""
    stats = ["mean", "std", "min", "max", "delta_mean_abs"]
    frame_names = BASE_FEATURE_NAMES + [f"mfcc_{index + 1}" for index in range(MFCC_COUNT)]
    return [f"{name}_{stat}" for stat in stats for name in frame_names]


def extract_speech_features(path: str | Path) -> np.ndarray:
    """Run speech preprocessing, feature extraction, and temporal modelling."""
    audio, source_rate = load_wav(path)
    audio = resample_linear(audio, source_rate, SAMPLE_RATE)
    audio = trim_silence(audio)
    frames = frame_audio(audio, SAMPLE_RATE)
    frame_features = extract_frame_features(frames, SAMPLE_RATE)
    return temporal_summary(frame_features)
