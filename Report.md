# Multimodal Emotion Recognition Report

## A. Architecture Decisions

### 1. Preprocessing

**Speech:** Audio is loaded from TESS `.wav` files, including a fallback for AIFF-formatted files stored with a `.wav` extension. The signal is converted to mono float audio, resampled to 16 kHz, silence-trimmed, and split into 25 ms frames with a 10 ms hop.

**Why:** Emotion in TESS is mainly expressed through vocal cues such as energy, pitch-related spectral structure, rhythm, and articulation. Normalizing sample rate and trimming silence makes feature extraction more consistent across files.

**Text:** Text is extracted from the spoken-word part of the filename, for example `OAF_base_angry.wav -> base`. The text is lowercased and tokenized into alphabetic tokens.

**Why:** The PDF requires a text-only pipeline, but TESS does not provide rich emotional sentences. The only dataset-aligned text is the spoken word in each file.

### 2. Feature Extraction

**Speech:** The system extracts RMS energy, zero-crossing rate, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, and MFCC-style mel cepstral coefficients.

**Why:** These features capture emotion-related acoustic cues: intensity, voice sharpness, spectral shape, and timbral changes.

**Text:** The TESS-trained text baseline uses word unigrams and character 2-gram/3-gram count features.

**Why:** Since each transcript is usually a single neutral word, character and word n-grams are the most reasonable lightweight text representation available from the dataset.

### 3. Temporal / Contextual Modelling

**Speech temporal modelling:** Frame-level speech features are summarized using mean, standard deviation, minimum, maximum, and mean absolute delta.

**Why:** These statistics model how emotional cues vary over the utterance while keeping the representation compact.

**Text contextual modelling:** The TESS text baseline uses bag-of-ngram counts. For live text inference, an external pretrained model, `j-hartmann/emotion-english-distilroberta-base`, is optionally used.

**Why:** TESS text alone is not emotionally informative. The external model is used only for user-entered emotional sentences in the UI and is reported separately from TESS-trained text results.

### 4. Fusion

**Architecture:** Early fusion is used by concatenating the speech temporal representation with the text n-gram representation. The text branch is assigned a tuned weight.

**Why:** The PDF asks for a unified multimodal representation. Early fusion is simple, transparent, and suitable for this small dataset. Since TESS emotion is mostly in the audio, the fusion model relies mainly on speech while still including text.

### 5. Classifier

**Speech-only:** Distance-weighted k-nearest-neighbors. The value of `k` is selected using a validation split.

**Text-only TESS baseline:** Multinomial Naive Bayes.

**Fusion:** Distance-weighted k-nearest-neighbors with tuned `k` and tuned text-branch weight.

**Why:** kNN performs well with compact acoustic representations on TESS, and validation tuning improved speech and fusion accuracy. Naive Bayes is appropriate for sparse text count features.

## B. Experiments

| Model Variant | Train Accuracy | Test Accuracy | Test Samples |
|---|---:|---:|---:|
| Speech-only | 100.00% | 97.32% | 560 |
| Text-only TESS baseline | 17.86% | 0.00% | 560 |
| Multimodal fusion | 100.00% | 97.14% | 560 |

The speech-only model performs best on the TESS test split. The fusion model is very close to speech-only, but does not improve because the text branch has little real emotional information in this dataset.

The external pretrained text model is used for live user-entered emotional sentences, but it is not counted as TESS-trained accuracy because it was not trained/evaluated on the TESS split.

## C. Analysis

### Easiest and Hardest Emotions

Using the speech-only test classification report:

| Emotion | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| angry | 1.00 | 1.00 | 1.00 |
| fear | 1.00 | 1.00 | 1.00 |
| disgust | 1.00 | 0.99 | 0.99 |
| neutral | 0.95 | 1.00 | 0.98 |
| sad | 1.00 | 0.95 | 0.97 |
| happy | 0.92 | 0.96 | 0.94 |
| pleasant_surprise | 0.95 | 0.91 | 0.93 |

**Easiest emotions:** `angry` and `fear`, both with F1 score of 1.00. These emotions tend to have stronger acoustic patterns such as higher intensity, sharper spectral changes, or more distinctive delivery.

**Hardest emotion:** `pleasant_surprise`, with the lowest F1 score. It is often confused with `happy` because both can have high energy and bright vocal tone.

### When Fusion Helps Most

Fusion helps most when both modalities contain useful and complementary emotion information. In this project, the TESS transcript text is usually a neutral word such as `base`, `book`, or `chair`, so it does not add much emotional signal. Therefore, fusion mostly follows the speech model.

If the user enters a real emotional sentence such as “I am very sad today,” the external pretrained text model can classify the text emotion well. However, for the official TESS experiment, speech remains the dominant modality.

### Error Analysis

Representative failure cases from the held-out test predictions:

| Pipeline | Word | Actual Emotion | Predicted Emotion | Likely Reason |
|---|---|---|---|---|
| Speech | hurl | disgust | pleasant_surprise | Some acted disgust samples may have high energy or sharp spectral cues similar to surprise. |
| Speech | food | happy | pleasant_surprise | Happy and pleasant surprise are acoustically close, both often bright and energetic. |
| Speech | good | happy | pleasant_surprise | Positive high-energy delivery can overlap with surprise. |
| Speech | dime | pleasant_surprise | happy | Surprise and happiness have similar prosodic patterns in TESS. |
| Fusion | hurl | disgust | neutral | The fusion model still mainly follows speech, but weighted text/noisy local neighbors can shift borderline samples. |

The most common confusion is between `happy` and `pleasant_surprise`. This is expected because both emotions can share high pitch, high energy, and a positive vocal style.

The text-only TESS baseline fails because the same transcript word appears under multiple emotion labels. For example, `base` can appear as angry, sad, happy, neutral, and other emotions. Text alone therefore cannot reliably determine the label.

### Representation Separability Visualizations

The project visualizes learned/derived representations using 2D PCA plots saved in `Results/plots/`:

| Block | Plot File |
|---|---|
| Temporal Modelling block | `Results/plots/speech_temporal_representation.svg` |
| Contextual Modelling block | `Results/plots/text_contextual_representation.svg` |
| Fusion block | `Results/plots/fusion_representation.svg` |

Interpretation:

**Speech temporal representation:** Shows the clearest emotion separability because vocal features carry the real emotional signal in TESS.

**Text contextual representation:** Shows weak separability because transcript words are neutral and repeated across emotions.

**Fusion representation:** Closely follows the speech representation because the text branch is weak and therefore down-weighted.

## Summary

The final system satisfies the three required variants: speech-only, text-only, and multimodal fusion. Speech-only and fusion perform strongly on TESS, while the TESS-trained text baseline is intentionally reported honestly as weak because the dataset does not contain emotionally meaningful text. An external pretrained text emotion model is included for live user-entered text inference, but it is documented separately from TESS-trained results.
