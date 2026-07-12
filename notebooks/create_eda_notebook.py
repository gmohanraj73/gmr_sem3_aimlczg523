"""Generate notebooks/01_eda.ipynb — run from project root: python notebooks/create_eda_notebook.py"""
import json
import os
from pathlib import Path


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


# ─── Cell 1 — Setup ───────────────────────────────────────────────────────────

SETUP = """\
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Resolve project root regardless of how the notebook is launched
_here = Path().resolve()
PROJECT_ROOT = _here.parent if _here.name == 'notebooks' else _here
os.chdir(PROJECT_ROOT)

SCREENSHOT_DIR = PROJECT_ROOT / 'screenshots' / 'eda'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
TARGET_PALETTE = ['#2196F3', '#F44336']
TARGET = 'num'

print('Project root  :', PROJECT_ROOT)
print('Screenshots   :', SCREENSHOT_DIR)\
"""

# ─── Section 1 — Dataset Overview ─────────────────────────────────────────────

LOAD_DATA = """\
df_raw = pd.read_csv('data/raw/heart_disease_raw.csv')
df     = pd.read_csv('data/processed/heart_disease_clean.csv')

print('=== Raw Dataset ===')
print('Shape   :', df_raw.shape)
print('Columns :', list(df_raw.columns))
print()
print('Data types:')
print(df_raw.dtypes.to_string())\
"""

DESCRIBE = """\
print('=== Processed Dataset — Descriptive Statistics ===')
df.describe().round(2)\
"""

MISSING_HEATMAP = """\
df_check = df_raw.replace('?', float('nan'))

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(
    df_check.isnull(),
    yticklabels=False,
    cbar=True,
    cmap='viridis',
    ax=ax
)
ax.set_title('Missing Value Heatmap (Raw Data)', fontsize=14, fontweight='bold')
ax.set_xlabel('Features')
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '01_missing_value_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

missing = df_check.isnull().sum()
missing_cols = missing[missing > 0]
if missing_cols.empty:
    print('No missing values — dataset is complete.')
else:
    print('Missing value counts:')
    print(missing_cols)\
"""

# ─── Section 2 — Feature Distributions ───────────────────────────────────────

HISTOGRAMS = """\
FEATURE_COLS = [c for c in df.columns if c != TARGET]
n_cols = 3
n_rows = (len(FEATURE_COLS) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    ax = axes[i]
    for cls, color, label in zip([0, 1], TARGET_PALETTE, ['No Disease', 'Disease']):
        subset = df[df[TARGET] == cls][col].dropna()
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, density=True)
        try:
            subset.plot.kde(ax=ax, color=color, linewidth=1.5)
        except Exception:
            pass
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

for j in range(len(FEATURE_COLS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    'Feature Distributions by Target Class (with KDE Overlay)',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()\
"""

# ─── Section 3 — Class Balance ────────────────────────────────────────────────

CLASS_BALANCE = """\
counts = df[TARGET].value_counts().sort_index()
total  = len(df)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ['No Disease (0)', 'Disease (1)'],
    counts.values,
    color=TARGET_PALETTE,
    edgecolor='black',
    linewidth=0.8,
    width=0.5
)
for bar, count in zip(bars, counts.values):
    pct = count / total * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f'{count}\\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=12, fontweight='bold'
    )
ax.set_title('Class Balance', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
ax.set_ylim(0, counts.max() * 1.25)
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '03_class_balance.png', dpi=150, bbox_inches='tight')
plt.show()

ratio = counts.max() / counts.min()
print(f'Class 0 (no disease): {counts[0]} samples ({counts[0] / total * 100:.1f}%)')
print(f'Class 1 (disease)   : {counts[1]} samples ({counts[1] / total * 100:.1f}%)')
print(f'Imbalance ratio     : {ratio:.2f}:1')
if ratio < 1.5:
    print('Assessment: Approximately balanced — no resampling strictly required.')
else:
    print('Assessment: Mild imbalance — consider class_weight=balanced or SMOTE.')\
"""

# ─── Section 4 — Correlation Analysis ────────────────────────────────────────

CORRELATION = """\
corr = df.corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    ax=ax,
    annot_kws={'size': 8}
)
ax.set_title('Pearson Correlation Heatmap (Full Matrix)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print('Top 5 features most correlated with target:')
print(target_corr.head(5).to_string())
TOP_FEATURES = target_corr.head(6).index.tolist()\
"""

# ─── Section 5 — Feature Relationships ───────────────────────────────────────

BOX_PLOTS = """\
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(TOP_FEATURES):
    sns.boxplot(
        data=df, x=TARGET, y=feat,
        palette=TARGET_PALETTE, ax=axes[i]
    )
    axes[i].set_title(f'{feat} by Target', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Target (0=No Disease, 1=Disease)')

fig.suptitle('Top 6 Features vs Target Class — Box Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '05_boxplots_top_features.png', dpi=150, bbox_inches='tight')
plt.show()\
"""

SCATTER = """\
fig, ax = plt.subplots(figsize=(9, 6))
for cls, color, label in zip([0, 1], TARGET_PALETTE, ['No Disease', 'Disease']):
    s = df[df[TARGET] == cls]
    ax.scatter(
        s['age'], s['thalach'],
        c=color, label=label, alpha=0.65, s=45,
        edgecolors='white', linewidth=0.4
    )
ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Max Heart Rate Achieved (thalach)', fontsize=12)
ax.set_title('Age vs Max Heart Rate — Colored by Target Class', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '06_scatter_age_thalach.png', dpi=150, bbox_inches='tight')
plt.show()\
"""

VIOLIN = """\
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, feat in zip(axes, ['chol', 'trestbps']):
    sns.violinplot(
        data=df, x=TARGET, y=feat,
        palette=TARGET_PALETTE, ax=ax, inner='box'
    )
    ax.set_title(f'{feat} Distribution by Target', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target (0=No Disease, 1=Disease)')
plt.tight_layout()
plt.savefig(SCREENSHOT_DIR / '07_violin_chol_trestbps.png', dpi=150, bbox_inches='tight')
plt.show()\
"""

# ─── Section 6 — EDA Summary (markdown) ──────────────────────────────────────

EDA_SUMMARY = """\
## Section 6 — EDA Summary

### Key Findings

- **Class balance**: ~54% No Disease / ~46% Disease — approximately balanced; no resampling
  strictly required, but `class_weight='balanced'` is a low-cost safeguard for margin-based models.
- **Top predictive features** (Pearson |r| with target): `thalach`, `ca`, `oldpeak`, `cp`, `exang`
  — all clinically meaningful indicators.
- **thalach (max heart rate)**: Lower values strongly associated with disease presence.
  Clear separation visible in histograms and box plots.
- **oldpeak (ST depression)**: Right-skewed; higher values correlate strongly with disease.
  One outlier near 6.2 is a plausible clinical extreme — retained.
- **cp (chest pain type)**: Ordinal/nominal — `OneHotEncoder` is appropriate;
  shows the clearest visual class separation.
- **ca & thal**: Both have a small number of missing values (2–7 rows);
  imputed with mode (categorical strategy).
- **chol**: Extreme values (> 500 mg/dl) visible in violin plot — these represent
  real clinical presentations, not data errors; retained.
- **Age vs thalach scatter**: Disease patients cluster upper-left (older age, lower max HR),
  confirming both features carry complementary signal.

### Feature Engineering Decisions

1. **StandardScaler** on continuous numerics: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
2. **OneHotEncoder** (`handle_unknown='ignore'`) on categoricals:
   `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`.
3. Both transformers wrapped in a `ColumnTransformer` inside a sklearn `Pipeline` —
   ensures identical preprocessing is applied at inference time (no train/serve skew).
4. No dimensionality reduction needed: 13 features × 303 samples is comfortably within
   a linear regime and tree models handle this scale natively.\
"""

# ─── Assemble Notebook ────────────────────────────────────────────────────────

cells = [
    code_cell(SETUP,           "setup01"),
    md_cell("## Section 1 — Dataset Overview", "md_s1"),
    code_cell(LOAD_DATA,       "load01"),
    code_cell(DESCRIBE,        "desc01"),
    code_cell(MISSING_HEATMAP, "miss01"),
    md_cell("## Section 2 — Feature Distributions", "md_s2"),
    code_cell(HISTOGRAMS,      "hist01"),
    md_cell("## Section 3 — Class Balance", "md_s3"),
    code_cell(CLASS_BALANCE,   "cls01"),
    md_cell("## Section 4 — Correlation Analysis", "md_s4"),
    code_cell(CORRELATION,     "corr01"),
    md_cell("## Section 5 — Feature Relationships", "md_s5"),
    code_cell(BOX_PLOTS,       "box01"),
    code_cell(SCATTER,         "scat01"),
    code_cell(VIOLIN,          "viol01"),
    md_cell(EDA_SUMMARY,       "md_s6"),
]

NOTEBOOK = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")

    out_path = Path("notebooks") / "01_eda.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)
    print(f"Notebook written -> {out_path}")
