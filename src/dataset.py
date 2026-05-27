"""Dataset metadata preparation for the TESS emotion recognition project."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_DIR = Path(
    r"C:\Users\sathw\Desktop\IIITH\TESS Toronto emotional speech set data"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "metadata.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def normalize_emotion(raw_emotion: str) -> str:
    """Normalize TESS label variants into one consistent label set."""
    emotion = raw_emotion.strip().lower()
    if emotion in {"ps", "pleasant_surprise", "pleasant_surprised"}:
        return "pleasant_surprise"
    return emotion


def parse_tess_filename(path: Path) -> dict[str, str]:
    """Extract speaker, text, and emotion fields from a TESS wav path."""
    stem_parts = path.stem.split("_")
    folder_parts = path.parent.name.split("_", maxsplit=1)
    if len(stem_parts) < 2 or len(folder_parts) < 2:
        raise ValueError(f"Unexpected TESS filename format: {path.name}")

    speaker = stem_parts[0].upper()
    emotion = normalize_emotion(folder_parts[1])
    text = stem_parts[1].lower()

    return {
        "speech_path": str(path),
        "speaker": speaker,
        "text": text,
        "emotion": emotion,
    }


def find_top_level_wavs(dataset_dir: Path) -> list[Path]:
    """Return wav files from direct emotion folders, skipping nested duplicates."""
    wav_paths: list[Path] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        wav_paths.extend(sorted(child.glob("*.wav")))
    return wav_paths


def build_metadata(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build metadata with a shared train/test split for all pipelines."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records = [parse_tess_filename(path) for path in find_top_level_wavs(dataset_dir)]
    if not records:
        raise ValueError(f"No top-level .wav files found in: {dataset_dir}")

    metadata = pd.DataFrame(records)
    rng = random.Random(random_state)
    test_indices = []
    for _, label_group in metadata.groupby("emotion"):
        indices = list(label_group.index)
        rng.shuffle(indices)
        test_count = round(len(indices) * test_size)
        test_indices.extend(indices[:test_count])

    metadata["split"] = "train"
    metadata.loc[test_indices, "split"] = "test"
    metadata = metadata.sort_values(["split", "emotion", "speaker", "text"]).reset_index(
        drop=True
    )
    return metadata


def save_metadata(metadata: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Save metadata as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TESS metadata CSV.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the top-level TESS dataset directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the generated metadata CSV.",
    )
    args = parser.parse_args()

    metadata = build_metadata(dataset_dir=args.dataset_dir)
    save_metadata(metadata, args.output_path)
    print(f"Saved {len(metadata)} rows to {args.output_path}")
    print(metadata.groupby(["split", "emotion"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
