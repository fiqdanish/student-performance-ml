# Student Performance Classification (SVM)

A data mining project that predicts student-related outcomes from academic and
background factors using Support Vector Machine (SVM) classifiers. Built for the
**Year 2 Sem 2 Data Mining** course (Project 1 — Classification).

## Overview

Student success is shaped by many overlapping factors — study hours, attendance,
parental involvement, access to resources, motivation, and more. This project
turns the question *"what drives student outcomes?"* into a measurable
classification task by cleaning the data, transforming it into model-ready
features, and training SVM models to predict three different targets ("insights").

## Dataset

- **`StudentPerformance.csv`** — the raw dataset.
- **`modif_student_performance.csv`** — the cleaned and encoded dataset produced
  by the preprocessing step (used by all model scripts).

Features include study hours, attendance, previous scores, sleep hours, tutoring
sessions, physical activity, parental involvement, access to resources,
motivation level, family income, teacher quality, peer influence, and more.

## Project Structure

| File | Description |
|------|-------------|
| `Preprocessing.py` | Cleans, transforms, and encodes the raw data into `modif_student_performance.csv`. |
| `SVM (Insight 1).py` | Predicts **Exam Score** band, with a confusion-matrix heatmap. |
| `SVM (Insight 2).py` | Predicts **Motivation Level**. |
| `SVM (Insight 3).py` | Predicts **Access to Resources**. |
| `Confusion matrix heatmap (Insight 1).png` | Saved visualization for Insight 1. |

## Preprocessing Steps

1. **Data cleaning** — drop missing values and duplicates; remove invalid
   records (e.g. exam scores above 100 or below 0).
2. **Data transformation**
   - Continuous features (hours studied, attendance, previous scores, exam
     score, sleep hours, tutoring sessions, physical activity) are grouped into
     bands using **equal-width binning** (`pd.cut`).
   - **Nominal** categorical features (e.g. gender, internet access) are
     converted with **Label Encoding**.
   - **Ordinal** categorical features (e.g. parental involvement, motivation
     level) are converted with **Ordinal Encoding**.
3. The result is exported to `modif_student_performance.csv`.

## Modeling

- **Algorithm:** Support Vector Machine (`SVC`, RBF kernel).
- **Configuration:** `C=1.0`, `gamma='scale'`, `class_weight='balanced'` to
  handle class imbalance.
- **Split:** 80/20 train/test with stratification (`random_state=42`).
- **Scaling:** features standardized with `StandardScaler`.
- **Evaluation:** accuracy, precision, recall, weighted F1 score, classification
  report, and confusion matrix (plus a heatmap for Insight 1).

## Requirements

- Python 3.11+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

1. **Preprocess the data** (generates `modif_student_performance.csv`):

   ```bash
   python "Preprocessing.py"
   ```

2. **Run any insight model:**

   ```bash
   python "SVM (Insight 1).py"
   python "SVM (Insight 2).py"
   python "SVM (Insight 3).py"
   ```

Each model script prints its evaluation metrics to the console; Insight 1 also
displays a confusion-matrix heatmap.

## Notes & Limitations

- Targets are **imbalanced**, which caps accuracy on minority classes; this is
  mitigated with balanced class weights and stratified splitting.
- Some features have **weak correlation** with the targets, limiting predictive
  power regardless of model tuning.
- Possible improvements: compare against other classifiers (Random Forest,
  Logistic Regression), tune hyperparameters with `GridSearchCV`, use
  cross-validation, and apply feature selection or SMOTE.
