"""Generate final result tables, plots, and report for the project."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion_features import combine_features
from src.speech_features import extract_speech_features
from src.text_features import TextVectorizer


RESULTS_DIR = PROJECT_ROOT / "Results"
PLOTS_DIR = RESULTS_DIR / "plots"


COLORS = {
    "angry": "#d73027",
    "disgust": "#1a9850",
    "fear": "#762a83",
    "happy": "#fdae61",
    "neutral": "#4575b4",
    "pleasant_surprise": "#66c2a5",
    "sad": "#313695",
}


def pca_2d(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def save_svg_scatter(points: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    width = 900
    height = 640
    margin = 70
    x = points[:, 0]
    y = points[:, 1]
    x_range = max(float(x.max() - x.min()), 1e-8)
    y_range = max(float(y.max() - y.min()), 1e-8)
    sx = margin + (x - x.min()) / x_range * (width - 2 * margin)
    sy = height - margin - (y - y.min()) / y_range * (height - 2 * margin)

    circles = []
    for px, py, label in zip(sx, sy, labels):
        circles.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" '
            f'fill="{COLORS[label]}" fill-opacity="0.72" />'
        )

    legend = []
    for index, label in enumerate(sorted(set(labels))):
        y_pos = 42 + index * 24
        legend.append(
            f'<rect x="690" y="{y_pos - 10}" width="14" height="14" '
            f'fill="{COLORS[label]}" />'
            f'<text x="712" y="{y_pos + 2}" font-size="14">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{margin}" y="38" font-size="22" font-family="Arial" font-weight="700">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>
<text x="{width / 2 - 30:.0f}" y="{height - 22}" font-size="14" font-family="Arial">PC1</text>
<text x="22" y="{height / 2:.0f}" font-size="14" font-family="Arial" transform="rotate(-90 22,{height / 2:.0f})">PC2</text>
{''.join(circles)}
{''.join(legend)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def combine_accuracy_tables() -> pd.DataFrame:
    rows = []
    for name in ("speech", "text", "fusion"):
        for split in ("train", "test"):
            path = RESULTS_DIR / f"{name}_{split}_accuracy.csv"
            if path.exists():
                rows.append(pd.read_csv(path))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(RESULTS_DIR / "all_model_accuracy_table.csv", index=False)
    return table


def generate_plots(metadata: pd.DataFrame) -> None:
    test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    labels = test_metadata["emotion"].astype(str).tolist()

    speech = np.vstack([extract_speech_features(path) for path in test_metadata["speech_path"]])
    save_svg_scatter(
        pca_2d(speech),
        labels,
        "Temporal Speech Representation",
        PLOTS_DIR / "speech_temporal_representation.svg",
    )

    text_vocab = json.loads(
        (PROJECT_ROOT / "models" / "text_pipeline" / "text_model.json").read_text(
            encoding="utf-8"
        )
    )["vocabulary"]
    text_vectorizer = TextVectorizer()
    text_vectorizer.vocabulary = text_vocab
    text = text_vectorizer.transform(test_metadata["text"].tolist())
    save_svg_scatter(
        pca_2d(text),
        labels,
        "Contextual Text Representation",
        PLOTS_DIR / "text_contextual_representation.svg",
    )

    fusion_vocab = json.loads(
        (PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_vocabulary.json").read_text(
            encoding="utf-8"
        )
    )
    fusion_vectorizer = TextVectorizer()
    fusion_vectorizer.vocabulary = fusion_vocab
    fusion_text = fusion_vectorizer.transform(test_metadata["text"].tolist())
    fusion = combine_features(speech, fusion_text)
    save_svg_scatter(
        pca_2d(fusion),
        labels,
        "Fusion Representation",
        PLOTS_DIR / "fusion_representation.svg",
    )


def easiest_and_hardest(report: pd.DataFrame) -> tuple[str, str]:
    ordered = report.sort_values("f1_score")
    return ordered.iloc[-1]["emotion"], ordered.iloc[0]["emotion"]


def generate_report(accuracy_table: pd.DataFrame) -> None:
    speech_report = pd.read_csv(RESULTS_DIR / "speech_test_classification_report.csv")
    fusion_report = pd.read_csv(RESULTS_DIR / "fusion_test_classification_report.csv")
    easiest, hardest = easiest_and_hardest(speech_report)

    test_rows = accuracy_table[accuracy_table["split"] == "test"]
    accuracy_lines = "\n".join(
        f"- {row.model_variant}: {row.accuracy:.4f} on {int(row.samples)} test samples"
        for row in test_rows.itertuples(index=False)
    )

    report = f"""# Multimodal Emotion Recognition Report

## A. Architecture Decisions

### Speech-only

Preprocessing loads the audio, handles RIFF WAV and AIFF-formatted files with `.wav`
extensions, converts to mono float audio, resamples to 16 kHz, trims silence, and
frames audio into 25 ms windows with a 10 ms hop.

Feature extraction computes frame-level RMS energy, zero-crossing rate, spectral
centroid, spectral bandwidth, spectral rolloff, spectral flatness, and
MFCC-style mel cepstral coefficients. Temporal modelling summarizes those frame
features with mean, standard deviation, minimum, maximum, and mean absolute
delta. The classifier is distance-weighted k-nearest-neighbors. `k` is selected
on a validation slice of the training data before retraining on the full train
split.

### Text-only

Text preprocessing uses the transcript word derived from each TESS filename,
lowercases it, and keeps alphabetic tokens. Feature extraction uses word unigrams
and character 2-grams/3-grams. Contextual modelling is represented by this compact
ngram count vector. The classifier is multinomial Naive Bayes.

The TESS transcript words are semantically neutral and repeated across emotions,
so text alone has little usable emotional information. The implementation avoids
using emotion labels embedded in filenames as text features because that would
leak the answer.

### Multimodal Fusion

Fusion uses early concatenation of the temporal speech representation and the text
ngram representation. The text branch is down-weighted in the fused distance
space because the dataset text content is weak for emotion classification. The
classifier is distance-weighted k-nearest-neighbors, with both `k` and text
branch weight selected on a validation slice of the training data.

## B. Experiments

{accuracy_lines}

## C. Analysis

Using the speech-only test report, the easiest emotion is `{easiest}` and the
hardest emotion is `{hardest}` by F1 score.

Fusion helps most when the text branch contributes complementary information. In
this TESS setup, the transcript is usually a neutral word such as `back`, `base`,
or `book`, so fusion mostly follows the speech representation and does not exceed
speech-only accuracy.

Error analysis can be performed from `Results/speech_test_predictions.csv`,
`Results/text_test_predictions.csv`, and `Results/fusion_test_predictions.csv`.
The most likely failures are confusions between acoustically similar acted
emotions, especially where pitch, energy, or spectral shape overlap.

Representation separability plots are saved in `Results/plots/`:

- `speech_temporal_representation.svg`
- `text_contextual_representation.svg`
- `fusion_representation.svg`

## Deliverables

The repository contains train/test scripts for all three variants, generated
accuracy tables, classification reports, confusion matrices, predictions, plots,
and this report.
"""
    (PROJECT_ROOT / "Report.md").write_text(report, encoding="utf-8")


def main() -> None:
    metadata = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
    accuracy_table = combine_accuracy_tables()
    generate_plots(metadata)
    generate_report(accuracy_table)
    print(f"Saved {RESULTS_DIR / 'all_model_accuracy_table.csv'}")
    print(f"Saved plots to {PLOTS_DIR}")
    print(f"Saved {PROJECT_ROOT / 'Report.md'}")


if __name__ == "__main__":
    main()
