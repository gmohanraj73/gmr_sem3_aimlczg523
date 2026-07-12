"""Generate notebooks/02_training.ipynb — run from project root:
    python notebooks/create_training_notebook.py

The notebook mirrors src/train.py (feature engineering + model development)
with markdown narration and inline plots, ending in a model-selection
justification cell.
"""
import json
import uuid
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }


TITLE = """\
# 02 — Feature Engineering & Model Development

**Heart Disease Risk Classifier · BITS Pilani AIMLCZG523 · Assignment 01**

This notebook mirrors `src/train.py`. It:

1. Loads the cleaned dataset produced by `src/data_processing.py`.
2. Builds a `ColumnTransformer` (scaling + one-hot encoding).
3. Trains and tunes **three** models — Logistic Regression, Random Forest,
   and XGBoost — each as a full sklearn `Pipeline`.
4. Compares them on the held-out test set and selects a winner by ROC-AUC.

> Every model is wrapped in a `Pipeline(preprocessor + estimator)` so the saved
> artifact is self-contained and reusable at inference time."""

SETUP = """\
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve project root regardless of launch directory.
_here = Path().resolve()
PROJECT_ROOT = _here.parent if _here.name == 'notebooks' else _here
os.chdir(PROJECT_ROOT)

plt.style.use('seaborn-v0_8-whitegrid')
print('Project root:', PROJECT_ROOT)"""

IMPORTS = """\
from src.data_processing import NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL, build_preprocessor
from src.train import build_search_spaces, evaluate

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
print('Numeric    :', NUMERIC_COLS)
print('Categorical:', CATEGORICAL_COLS)
print('Target     :', TARGET_COL)"""

LOAD = """\
df = pd.read_csv('data/processed/heart_disease_clean.csv')
X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].astype(int)
print('Feature matrix:', X.shape)
print('Class balance :', dict(y.value_counts().sort_index()))
X.head()"""

SPLIT = """\
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print('Train:', X_train.shape[0], ' Test:', X_test.shape[0])"""

PREPROC = """\
preprocessor = build_preprocessor(X)
preprocessor"""

TRAIN = """\
searches = build_search_spaces(preprocessor)

fitted = {}
best_params_all = {}
results = {}

for name, (search, _, tuner) in searches.items():
    print(f'Tuning {name} via {tuner} ...')
    search.fit(X_train, y_train)
    fitted[name] = search.best_estimator_
    best_params_all[name] = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
    results[name] = evaluate(fitted[name], X_train, y_train, X_test, y_test)
    print(f'   best params : {best_params_all[name]}')
    print(f'   test ROC-AUC: {results[name][\"roc_auc\"]}')"""

COMPARE = """\
comparison = pd.DataFrame(results).T[
    ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_mean', 'cv_std']
].round(4)
comparison"""

ROC = """\
from sklearn.metrics import roc_curve, roc_auc_score

fig, ax = plt.subplots(figsize=(8, 7))
palette = {'LogisticRegression': '#2196F3', 'RandomForest': '#4CAF50', 'XGBoost': '#F44336'}
for name, pipe in fitted.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, proba):.4f})',
            color=palette[name], lw=2)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Heart Disease Classifiers'); ax.legend(loc='lower right')
plt.show()"""

CM = """\
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, pipe) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name)
fig.suptitle('Confusion Matrices (Test Set)'); plt.tight_layout(); plt.show()"""

FI = """\
tree_models = [n for n in ('RandomForest', 'XGBoost') if n in fitted]
fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 6))
axes = np.atleast_1d(axes)
for ax, name in zip(axes, tree_models):
    pipe = fitted[name]
    feat_names = pipe.named_steps['preprocessor'].get_feature_names_out()
    importances = pipe.named_steps['model'].feature_importances_
    order = np.argsort(importances)[::-1][:15]
    ax.barh([feat_names[i] for i in order][::-1], importances[order][::-1], color='#673AB7')
    ax.set_title(f'{name} — Top Feature Importances'); ax.set_xlabel('Importance')
plt.tight_layout(); plt.show()"""

WINNER = """\
best_name = max(results, key=lambda n: results[n]['roc_auc'])
print('Winner:', best_name, '(ROC-AUC =', results[best_name]['roc_auc'], ')')
print('Best params:', best_params_all[best_name])"""

JUSTIFY = """\
## Model Selection Justification

The winning model is chosen by **test-set ROC-AUC**, which is the most
appropriate headline metric here because:

- **Clinical framing** — this is a screening task; ranking patients by risk
  (which ROC-AUC measures) matters more than accuracy at a single threshold.
- **Threshold independence** — ROC-AUC summarises performance across all
  decision thresholds, so the choice is robust to where we later set the
  0.5 cutoff.
- **Class balance** — the dataset is only mildly imbalanced, and ROC-AUC is
  read alongside precision/recall/F1 in the comparison table above to confirm
  the winner is not trading recall for precision (or vice-versa).

The winner's 5-fold cross-validated ROC-AUC (`cv_mean ± cv_std`) is reported
next to its test score to confirm the result generalises and is not an artifact
of the particular train/test split. The full fitted pipeline (preprocessor +
estimator) for this model is what `src/train.py` serializes to
`models/best_model_pipeline.pkl`."""

NOTE_IMPORTS_FIX = None  # placeholder


def build() -> dict:
    cells = [
        md_cell(TITLE),
        md_cell("## 1 — Setup"),
        code_cell(SETUP),
        code_cell(IMPORTS),
        md_cell("## 2 — Load Processed Data"),
        code_cell(LOAD),
        md_cell("## 3 — Train/Test Split (stratified 80/20)"),
        code_cell(SPLIT),
        md_cell("## 4 — Preprocessing Pipeline"),
        code_cell(PREPROC),
        md_cell(
            "## 5 — Train & Tune Three Models\n\n"
            "Logistic Regression (GridSearchCV), Random Forest and XGBoost "
            "(RandomizedSearchCV), each a full `Pipeline`."
        ),
        code_cell(TRAIN),
        md_cell("## 6 — Model Comparison"),
        code_cell(COMPARE),
        md_cell("## 7 — ROC Curves"),
        code_cell(ROC),
        md_cell("## 8 — Confusion Matrices"),
        code_cell(CM),
        md_cell("## 9 — Feature Importances"),
        code_cell(FI),
        md_cell("## 10 — Winner"),
        code_cell(WINNER),
        md_cell(JUSTIFY),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "02_training.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print("Wrote", out)
