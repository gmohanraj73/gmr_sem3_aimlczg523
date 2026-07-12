"""Heart Disease UCI dataset — loading, cleaning, and processing utilities."""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL: str = "num"


class DataProcessor:
    """Handles loading, cleaning, encoding, and saving the Heart Disease dataset."""

    def load_raw(self, path: str) -> pd.DataFrame:
        """Load raw CSV from disk."""
        return pd.read_csv(path)

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace '?' with NaN, then impute: numeric → median, categorical → mode."""
        df = df.copy()
        # Opt into pandas' future non-downcasting behavior to avoid a
        # FutureWarning; numeric columns are explicitly cast just below anyway.
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace("?", np.nan)

        # Force numeric dtype (may be object when '?' was present in CSV)
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Median imputation for numeric columns
        for col in NUMERIC_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Mode imputation for categorical columns
        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binarize target: 0 = no disease, 1 = disease present (any value > 0)."""
        df = df.copy()
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
        return df

    def get_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Return dict with 'numeric' and 'categorical' column name lists."""
        present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
        present_categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
        return {"numeric": present_numeric, "categorical": present_categorical}

    def save_processed(self, df: pd.DataFrame, path: str) -> None:
        """Save cleaned DataFrame as CSV, creating parent directories if needed."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Processed data saved -> {path}")


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the feature matrix.

    - Numeric columns  -> StandardScaler (zero mean, unit variance)
    - Categorical cols -> OneHotEncoder (dense, unknown categories ignored)

    Only columns actually present in ``df`` are wired in, so the same helper
    works on both the full dataset and any subset passed at inference time.
    """
    feature_types = DataProcessor().get_feature_types(df)
    numeric = feature_types["numeric"]
    categorical = feature_types["categorical"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def run_pipeline() -> None:
    """Run the full data processing pipeline end to end."""
    processor = DataProcessor()

    raw_path = "data/raw/heart_disease_raw.csv"
    processed_path = "data/processed/heart_disease_clean.csv"

    print("Loading raw data...")
    df_raw = processor.load_raw(raw_path)
    print(f"Raw shape     : {df_raw.shape}")
    print(f"Missing (NaN) : {df_raw.isnull().sum().sum()} total")

    print("\nHandling missing values...")
    df_clean = processor.handle_missing(df_raw)

    print("Encoding target column...")
    df_clean = processor.encode_target(df_clean)

    feature_types = processor.get_feature_types(df_clean)

    print("\n--- Before vs After ---")
    print(f"Shape  before : {df_raw.shape}  |  after : {df_clean.shape}")
    print(f"Missing before: {df_raw.isnull().sum().sum()}  |  after : {df_clean.isnull().sum().sum()}")
    print(f"\nClass distribution:\n{df_clean[TARGET_COL].value_counts().sort_index()}")
    print(f"\nNumeric features    : {feature_types['numeric']}")
    print(f"Categorical features: {feature_types['categorical']}")

    processor.save_processed(df_clean, processed_path)


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
    run_pipeline()
