# Multimodal Emotion Recognition

This project follows the provided problem statement for recognizing emotions
from speech-only input, text-only input, and combined speech-text multimodal
input using the Toronto Emotional Speech Set (TESS).

## Dataset

Expected local dataset path:

```text
C:\Users\sathw\Desktop\IIITH\TESS Toronto emotional speech set data
```

The current dataset contains `.wav` files organized by emotion folders. Since
separate transcript files are not present, the text input will be derived from
the filename word, for example:

```text
OAF_back_angry.wav -> text: back, emotion: angry
```

Only one dataset level should be used to avoid duplicate samples from the
nested duplicate folder.

## Required Functional Blocks

1. Preprocessing
2. Feature extraction
3. Temporal/contextual modelling
4. Fusion
5. Classifier

## Project Structure

```text
project/
|-- models/
|   |-- speech_pipeline/
|   |   |-- train.py
|   |   `-- test.py
|   |-- text_pipeline/
|   |   |-- train.py
|   |   `-- test.py
|   `-- fusion_pipeline/
|       |-- train.py
|       `-- test.py
|-- Results/
|   `-- plots/
|-- data/
|   `-- metadata.csv
|-- src/
|   `-- shared preprocessing, features, classifiers, inference
|-- frontend/
|   `-- prediction UI assets
|-- api.py
|-- run_all.py
|-- Report.md
|-- README.md
`-- requirements.txt
```

## Status

Step 1 is complete: project scaffold and dependency list.

Step 2 is complete: shared TESS metadata preparation.

- Uses 2800 top-level `.wav` files.
- Skips the nested duplicate dataset folder.
- Uses an 80/20 stratified train/test split.
- Extracts text from filenames and emotion labels from parent folders.
- Normalizes `pleasant_surprised`, `Pleasant_surprise`, and `ps` variants to
  `pleasant_surprise`.

Step 3 is complete: speech-only pipeline.

- Preprocessing: load mono PCM wav, resample to 16 kHz, trim silence, frame
  audio into 25 ms windows with 10 ms hop.
- Feature extraction: frame-level RMS, zero-crossing rate, spectral centroid,
  spectral bandwidth, spectral rolloff, spectral flatness, and MFCC-style mel
  cepstral features.
- Temporal modelling: utterance-level mean, standard deviation, minimum,
  maximum, and mean absolute delta statistics over frame features.
- Classifier: distance-weighted k-nearest-neighbors. `k` is selected on a
  validation split before retraining on the full train split.

Run:

```bash
python models/speech_pipeline/train.py
python models/speech_pipeline/test.py
```

Step 4 is complete: text-only pipeline.

- Preprocessing: lowercase transcript text and retain alphabetic tokens.
- Feature extraction/contextual modelling: word unigram and character 2/3-gram
  count features.
- Classifier: multinomial Naive Bayes.
- Optional external pretrained inference model:
  `j-hartmann/emotion-english-distilroberta-base`. This is used for live text
  inference when `torch` and `transformers` are installed. It is reported
  separately from the TESS-trained text baseline.

Step 5 is complete: multimodal fusion pipeline.

- Fusion: early concatenation of speech temporal features and text ngram
  features.
- Classifier: distance-weighted k-nearest-neighbors. `k` and text branch weight
  are selected on a validation split.

Step 6 is complete: non-UI inference API and final artifacts.

Step 7 is complete: responsive frontend UI integrated with the backend API.

- Prediction-focused UI for speech-only, text-only, and fusion model variants.
- Result display cards, loading states, validation, and API error handling.

## Setup

Use Python 3.12 or a compatible Python 3 environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

The live text predictor can use the pretrained Hugging Face model
`j-hartmann/emotion-english-distilroberta-base`. The first text prediction may
download model weights if they are not already cached.

## Dataset Setup

Place the TESS dataset beside this project folder:

```text
IIITH/
|-- project/
`-- TESS Toronto emotional speech set data/
```

The dataset folder should contain the emotion folders such as `OAF_angry`,
`YAF_sad`, etc.

## Run Full Workflow

```bash
python run_all.py
```

This regenerates metadata, trains/tests all three pipelines, and rebuilds final
tables, plots, and the report.

## Run Individual Pipelines

```bash
python models/speech_pipeline/train.py
python models/speech_pipeline/test.py
python models/text_pipeline/train.py
python models/text_pipeline/test.py
python models/fusion_pipeline/train.py
python models/fusion_pipeline/test.py
python src/artifacts.py
```

## API

Run the backend API and frontend website:

```bash
python api.py --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Endpoints:

```text
GET  /health
GET  /api/results
GET  /api/sample
POST /predict/speech  {"speech_path": "..."}
POST /predict/text    {"text": "back"}
POST /predict/fusion  {"speech_path": "...", "text": "back"}
POST /api/predict/text
POST /api/predict/upload
```

Upload prediction accepts `multipart/form-data` fields:

```text
mode: speech | text | fusion
text: transcript text
audio: .wav, .aiff, or .aif file
```

## Final Outputs

- `Results/all_model_accuracy_table.csv`
- `Results/*_classification_report.csv`
- `Results/*_confusion_matrix.csv`
- `Results/*_predictions.csv`
- `Results/plots/*.svg`
- `Report.md`
