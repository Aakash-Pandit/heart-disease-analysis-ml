# Heart Disease Classification — End-to-End ML Pipeline

A supervised binary classification system that predicts the presence of heart disease from clinical patient data. Built with a focus on model selection, hyperparameter optimization, and evaluation rigor.

---

## Problem Statement

Given 13 clinical features (demographics, vitals, ECG results), predict whether a patient has heart disease (`1`) or not (`0`). This is a binary classification problem where **recall is prioritized over precision** — a missed positive (false negative) carries higher clinical cost than a false alarm.

---

## Dataset

- **Source**: Cleveland Heart Disease Dataset
- **Samples**: 303 patients, no missing values
- **Features**: 13 (mix of continuous and categorical)
- **Target**: Binary — `1` = heart disease present, `0` = absent
- **Class balance**: 54.5% positive, 45.5% negative (near-balanced, no resampling needed)

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age in years |
| `sex` | binary | 0 = Female, 1 = Male |
| `cp` | categorical | Chest pain type (0–3) |
| `trestbps` | int | Resting blood pressure (mmHg) |
| `chol` | int | Serum cholesterol (mg/dl) |
| `fbs` | binary | Fasting blood sugar > 120 mg/dl |
| `restecg` | categorical | Resting ECG results (0–2) |
| `thalach` | int | Max heart rate achieved |
| `exang` | binary | Exercise-induced angina |
| `oldpeak` | float | ST depression (exercise vs rest) |
| `slope` | categorical | Slope of peak exercise ST segment |
| `ca` | int | Major vessels colored by fluoroscopy (0–4) |
| `thal` | categorical | Thalassemia type |

---

## ML Pipeline

```
Raw CSV → EDA → Feature/Target Split → Train/Test Split (80/20)
       → Model Training → Hyperparameter Tuning → Evaluation → Serialization → Inference
```

### Model Comparison

Three algorithms were trained and benchmarked before selecting the final model:

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression | **88.52%** | Selected — best accuracy + interpretability |
| Random Forest | 86.89% | Tuned with RandomizedSearchCV |
| K-Nearest Neighbors | 75.41% | Weak on this feature space |

### Hyperparameter Tuning

- **Stage 1 — RandomizedSearchCV**: Broad search over `C` and `solver` space (5-fold CV)
- **Stage 2 — GridSearchCV**: Refined search around best candidates from Stage 1
- **Final params**: `C=0.204`, `solver='liblinear'`

### Evaluation Metrics (Final Model)

| Metric | Score |
|---|---|
| Accuracy | 88.52% |
| Precision | 0.88 |
| Recall | **0.91** |
| F1-Score | 0.89 |
| Cross-validated Accuracy | 84.80% |
| Cross-validated Recall | 92.73% |

High recall (91%) is the key metric here. In a clinical screening context, failing to flag a sick patient is worse than over-flagging a healthy one.

**Confusion Matrix (Test Set — 61 samples):**
```
                  Predicted: No  Predicted: Yes
Actual: No              25             4
Actual: Yes              3            29
```

### On Confidence Scores

The model outputs calibrated probability scores rather than hard predictions. A 74% confidence on a borderline patient is intentional and correct behavior — it signals ambiguity in the feature space, not model failure. Overconfident models (99% on everything) are a reliability red flag in production systems.

---

## Project Structure

```
heart-disease-analysis-ml/
├── end-to-end-disease-classification.ipynb   # Full pipeline: EDA → training → inference
├── heart_disease_model.joblib                # Serialized final model (joblib)
├── data/
│   └── heart-disease.csv                     # Raw dataset
├── environment.yml                           # Conda environment spec
└── README.md
```

---

## Inference

Load the serialized model and pass a patient record as a dict:

```python
import joblib
import pandas as pd

model = joblib.load("heart_disease_model.joblib")

patient = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}

feature_order = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X = pd.DataFrame([patient])[feature_order]
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][pred]

print(f"Heart disease: {'Yes' if pred == 1 else 'No'} (confidence: {prob:.0%})")
```

---

## Tech Stack

- **Language**: Python 3.14
- **ML**: scikit-learn 1.8
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Serialization**: joblib
- **Environment**: Conda
