# Computing Social State from Facial Structure  
Term-time project- aprroach2 — Shuran Zhang  

This aprroach explores whether social behaviour can be computationally inferred from facial geometry, emotion labels, and contextual metadata.
It builds a pipeline from facial feature extraction to machine learning modelling.

# The Poster

- `Facial Datase Poster.pdf`


# The Dataset

## Dataset Components

### Raw images
- `dataset/`

### Manual annotations
- `dataset_label.xlsx`
  - emotion
  - context
  - behavioural annotations
  - social_state labels

### Processed files
- `features.csv`
- `dataset_with_features.csv`


# Code and Experiments

## Facial Feature Extraction

Script:

- `step2_extract_features_tasks.py`
This step converts visual data into machine-readable features.

## Social State Modelling

- `STEP 3 — Social State Modelling`
The goal is to test whether social behaviour can be computationally inferred from structured facial data.

Run:

```bash
python step2_extract_features_tasks.py
python step3_train_social_state.py
