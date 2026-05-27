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


## Setup

Use Python 3.12 or a compatible Python 3 environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

For optional local pretrained text-emotion inference, install:

```bash
pip install -r requirements-optional.txt
```

The optional live text predictor uses the pretrained Hugging Face model
`j-hartmann/emotion-english-distilroberta-base`. The first text prediction may
download model weights if they are not already cached. The Vercel deployment
uses the lightweight project dependencies for reliable deployment.

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



Upload prediction accepts `multipart/form-data` fields:

```text
mode: speech | text | fusion
text: transcript text
audio: .wav, .aiff, or .aif file
```

