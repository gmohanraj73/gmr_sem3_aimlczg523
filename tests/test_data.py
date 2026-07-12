"""Unit tests for the DataProcessor class in src/data_processing.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    DataProcessor,
)


@pytest.fixture
def processor() -> DataProcessor:
    return DataProcessor()


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic frame with '?' placeholders and missing values."""
    return pd.DataFrame(
        {
            "age": [63, 67, "?", 41, 56],
            "trestbps": [145, 160, 120, 130, "?"],
            "chol": [233, 286, 229, 204, 236],
            "thalach": [150, 108, 129, 172, 178],
            "oldpeak": [2.3, 1.5, 2.6, 1.4, 0.8],
            "sex": [1, 1, 1, 0, 1],
            "cp": [1, 4, 4, 3, 2],
            "fbs": [1, 0, 0, 0, 0],
            "restecg": [2, 2, 2, 0, 2],
            "exang": [0, 1, 1, 0, 0],
            "slope": [3, 2, 2, 3, 1],
            "ca": [0.0, 3.0, np.nan, 0.0, 0.0],
            "thal": [6.0, 3.0, 7.0, 3.0, 3.0],
            TARGET_COL: [0, 2, 1, 0, 3],
        }
    )


def test_load_raw_returns_dataframe(processor, tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame({"age": [1, 2], TARGET_COL: [0, 1]}).to_csv(csv, index=False)
    df = processor.load_raw(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_handle_missing_removes_nulls(processor, synthetic_df):
    cleaned = processor.handle_missing(synthetic_df)
    assert cleaned.isnull().sum().sum() == 0
    # '?' placeholders must be gone too.
    assert not (cleaned.astype(str) == "?").any().any()


def test_encode_target_is_binary(processor, synthetic_df):
    cleaned = processor.handle_missing(synthetic_df)
    encoded = processor.encode_target(cleaned)
    assert set(encoded[TARGET_COL].unique()).issubset({0, 1})


def test_feature_types_returns_correct_keys(processor, synthetic_df):
    types = processor.get_feature_types(synthetic_df)
    assert set(types.keys()) == {"numeric", "categorical"}
    assert set(types["numeric"]) == set(NUMERIC_COLS)
    assert set(types["categorical"]) == set(CATEGORICAL_COLS)


def test_processed_file_saved(processor, synthetic_df, tmp_path):
    out = tmp_path / "clean.csv"
    processor.save_processed(synthetic_df, str(out))
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert reloaded.shape[0] == synthetic_df.shape[0]


@pytest.mark.parametrize("bad_value", ["?", np.nan])
def test_handle_missing_handles_various_missing(processor, synthetic_df, bad_value):
    df = synthetic_df.copy()
    df["chol"] = df["chol"].astype(object)  # allow '?' / NaN without dtype warning
    df.loc[0, "chol"] = bad_value
    cleaned = processor.handle_missing(df)
    assert cleaned["chol"].isnull().sum() == 0
