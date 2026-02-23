# Term-time Project — Submission Guide  
## Computing Social State from Facial Structure  
Shuran Zhang  

---

# 1️⃣ The Poster

Submit the final version of the poster as a PDF.

Included file:

- `Facial Datase Poster.pdf`

All team members should submit the same poster file.

The poster presents:
- Dataset construction  
- Two research approaches  
- Machine learning modelling  
- Social interpretation findings  

---

# 2️⃣ The Dataset

Upload the dataset used for this project.  
If the dataset is too large for the VLE, provide a cloud storage link (e.g., OneDrive).

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

All team members submit the same dataset.

---

# 3️⃣ Code and Experiments

Submit all coding experiments and exploratory work conducted with the dataset.

## Facial Feature Extraction

Script:
- `step2_extract_features_tasks.py`

Run:

```bash
python step2_extract_features_tasks.py