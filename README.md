# Student Success Prediction with Logistic Regression

A supervised machine learning workflow built in Python to predict student pass/fail outcomes based on academic performance and engagement metrics.

## Overview
This project implements an end-to-end classification pipeline using **Logistic Regression**. It covers data ingestion, exploratory data analysis (EDA), missing value imputation/removal, target visualization, train-test splitting, model training, and performance evaluation.

## Dataset (`student_success_2000.csv`)
The dataset contains **2,000 student records** across **8 features**:

| Feature | Description | Data Type |
| :--- | :--- | :--- |
| `StudentID` | Unique identifier for each student | `int64` |
| `StudyHours` | Weekly study hours (1–20) | `int64` |
| `Attendance` | Class attendance percentage (40.0–100.0) | `float64` |
| `AssignmentScore` | Average assignment score (30.0–100.0) | `float64` |
| `TestScore` | Exam/test score (25–100) | `int64` |
| `Participation` | Class participation rating (1–10) | `int64` |
| `PreviousAverage` | Prior academic average (30–90) | `int64` |
| `Pass` | **Target variable** (`0` = Fail, `1` = Pass) | `int64` |

## Methodology
1. **Data Loading & Inspection**: Loaded `student_success_2000.csv` via Pandas and inspected dimensions (`(2000, 8)`), data types, and summary statistics.
2. **Data Cleaning**: Identified missing values in `Attendance` (40 nulls) and `AssignmentScore` (30 nulls) and dropped incomplete rows using `.dropna()`, resulting in **1,930 clean observations**.
3. **Exploratory Visualization**: Generated a bar chart with Matplotlib to visualize class distribution (`Pass` vs. `Fail`).
4. **Feature & Target Split**: Separated `Pass` as the target variable (`y`) and retained the remaining columns as predictors (`X`), then applied an **80/20 train-test split** (`random_state=42`).
5. **Model Training**: Trained a `LogisticRegression(max_iter=1000)` classifier using Scikit-Learn.
6. **Evaluation**: Assessed test set performance (`n = 386`) using accuracy, a confusion matrix, and a full classification report.

## Model Performance
Evaluated on the 20% holdout test set (386 students):

* **Overall Accuracy**: **84.72%** (`0.8471`)

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Fail)** | 0.83 | 0.84 | 0.83 | 177 |
| **1 (Pass)** | 0.86 | 0.86 | 0.86 | 209 |
| **Macro Avg** | 0.85 | 0.85 | 0.85 | 386 |
| **Weighted Avg** | 0.85 | 0.85 | 0.85 | 386 |

### Confusion Matrix
```text
[[148  29]   # True Negatives: 148 | False Positives: 29
 [ 30 179]]  # False Negatives: 30 | True Positives: 179
