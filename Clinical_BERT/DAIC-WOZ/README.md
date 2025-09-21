# Multimodal Emotion Recognition on DAIC-WOZ

This repository implements a multimodal emotion classification pipeline using the [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) dataset, combining **text** (transcripts) and **audio** (speech features).  
It leverages a `ClinicalBERT` model fine-tuned on [CounselChat](https://github.com/nbertagnolli/counsel-chat) for text-based emotion detection, and integrates prosodic features (MFCC, pitch, energy) for audio-based signals.

---

## Project Structure
- dacwoz_downloader.py # Download and Unzip the daicwoz data from URL
1. process_transcript.py # Merge all participant-only transcript
2. assign_labels.py # Assign emotion lavels to participant-only transcripts based on emotion schema
3. normalize_labels_with_ids.py # Build filename id to match audio clips and features
4. segment_audio.py # Segment participant-only audio data
5. extract_audio_features.py # Extract audio features: mfcc, putch, energy from the audio clips
6. merge_audio_text.py # Merge labeled transcripts data with the audio data
- Model training and evaluation
prep_multimodal.py
train_early_fusion.py # Train joint multimodal early-fusion model
train_late_fusion.py # Train separate text/audio models for late fusion
eval_late_fusion.py # Evaluate trained text/audio/fusion models
-
mm_train.parquet
mm_val.parquet
mm_test.parquet
mm_labels.json # label2id and id2label mappings
-
README.md


---

## Data Preprocessing

**Goal:** Convert DAIC-WOZ interviews into utterance-level data with aligned text, audio features, and emotion labels.

### 1. Transcript parsing
- Parse `.csv` transcripts into individual utterances
- Columns: `session_id`, `start_time`, `end_time`, `speaker`, `text`

### 2. Audio segmentation + feature extraction
- Segment `.wav` audio clips based on transcript timings
- Extract per-segment features:
  - **Prosody:** pitch (mean), energy (RMS)
  - **Spectral:** 13 MFCCs (mfcc_0 … mfcc_12)
- Normalize features (z-score across training set)

### 3. Merge with labels
- Align utterances with gold emotion labels
- Generate multi-hot `label_vec` for multi-label classification
- Save as:
  - `mm_train.parquet`, `mm_val.parquet`, `mm_test.parquet`
  - `mm_labels.json` (label2id and id2label mapping)

---

## Text Model: Fine-Tune ClinicalBERT on CounselChat

**Purpose:** Teach the text encoder to recognize mental health–related emotional content before multimodal training.

- Model: `emilyalsentzer/Bio_ClinicalBERT`
- Dataset: [CounselChat](https://github.com/nbertagnolli/counsel-chat)
- Training:
  - Reformulated as multi-label classification
  - Weighted binary cross-entropy loss (pos_weight)
  - Dynamic per-label threshold tuning
- Output:
  - `counselchat_weighted_model/best_model/` directory

This model is used as the text backbone for DAIC-WOZ multimodal training.

---

## Multimodal Models

### Early Fusion (`train_early_fusion.py`)
- Concatenates text [CLS] embedding with audio features
- Single joint classifier on fused representation
- Learns cross-modal correlations directly

### Late Fusion (`train_late_fusion.py`)

This setup trains the **text** and **audio** models **separately**, then combines their probabilities at inference.

What it does:
Trains a text classifier (ClinicalBERT fine-tuned on CounselChat, then adapted to DAIC-WOZ labels).
Trains an audio MLP on MFCC/pitch/energy features.

Saves:
mm_late_fusion/text/best_model/
mm_late_fusion/audio/best_model.pt
mm_late_fusion/thresholds.json (per-label thresholds, if available)

### Evaluation

Script: eval_late_fusion.py

Loads the best saved models

Tunes thresholds and α on the validation set

Evaluates on the test set

Run:
python eval_late_fusion.py

Outputs:

eval_reports/per_label_{text,audio,fusion}.csv — per-label precision, recall, F1, support

eval_reports/summary.json — micro/macro metrics and best α

eval_predictions/test_predictions.parquet — full predictions for error analysis

### Key Metrics

Micro-F1: overall accuracy weighted by label frequency

Macro-F1: balanced performance across all labels (rare and common)

Per-label reports: track which emotions benefit from multimodal fusion


