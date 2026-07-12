"""Download Heart Disease UCI dataset and save raw CSV with a summary report."""
from __future__ import annotations

import os

import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_save() -> None:
    print("Fetching Heart Disease UCI dataset (id=45)...")
    heart_disease = fetch_ucirepo(id=45)

    X: pd.DataFrame = heart_disease.data.features
    y: pd.DataFrame = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)

    # Detect '?' placeholders common in older UCI text files
    qmark_counts = (df.astype(str) == "?").sum()
    qmarks = qmark_counts[qmark_counts > 0]

    nan_counts = df.isnull().sum()
    nans = nan_counts[nan_counts > 0]

    print(f"\nDataset shape   : {df.shape}")
    print(f"Feature columns : {list(X.columns)}")
    print(f"Target column   : {list(y.columns)}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")

    print("\n--- Missing Value Report ---")
    if not qmarks.empty:
        print(f"'?' placeholders:\n{qmarks}")
    else:
        print("No '?' placeholders found.")

    if not nans.empty:
        print(f"NaN values:\n{nans}")
    else:
        print("No NaN values found.")

    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/heart_disease_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data saved  -> {raw_path}")

    target_col = y.columns[0]
    info_lines = [
        "Dataset       : Heart Disease UCI (id=45)",
        f"Shape         : {df.shape}",
        f"Columns       : {list(df.columns)}",
        "",
        "Data types:",
        df.dtypes.to_string(),
        "",
        "Missing values ('?'):",
        qmarks.to_string() if not qmarks.empty else "None",
        "",
        "Missing values (NaN):",
        nans.to_string() if not nans.empty else "None",
        "",
        "Class distribution (raw):",
        df[target_col].value_counts().sort_index().to_string(),
    ]
    info_path = "data/raw/data_info.txt"
    with open(info_path, "w") as f:
        f.write("\n".join(info_lines))
    print(f"Data info saved -> {info_path}")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "data":
        os.chdir("..")
    download_and_save()
