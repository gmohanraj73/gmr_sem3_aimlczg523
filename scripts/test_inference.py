"""Standalone inference smoke test — simulate a production model load.

Loads ``models/best_model_pipeline.pkl`` fresh (no training code involved),
runs predictions on a few sample patients, and asserts the artifact is a
self-contained pipeline (preprocessor + model) producing valid outputs.

Run from the project root:
    python scripts/test_inference.py
"""
from __future__ import annotations

import os
import sys

import joblib

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import HeartDiseasePredictor  # noqa: E402

MODEL_PATH = "models/best_model_pipeline.pkl"

# Three sample patients (raw feature dicts, same columns as the raw dataset).
SAMPLE_PATIENTS = [
    {
        "name": "Patient A",
        "features": {
            "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
            "slope": 3, "ca": 0, "thal": 6,
        },
    },
    {
        "name": "Patient B",
        "features": {
            "age": 37, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250,
            "fbs": 0, "restecg": 0, "thalach": 187, "exang": 0, "oldpeak": 3.5,
            "slope": 3, "ca": 0, "thal": 3,
        },
    },
    {
        "name": "Patient C",
        "features": {
            "age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 229,
            "fbs": 0, "restecg": 2, "thalach": 129, "exang": 1, "oldpeak": 2.6,
            "slope": 2, "ca": 2, "thal": 7,
        },
    },
]


def main() -> None:
    if os.path.basename(os.getcwd()) == "scripts":
        os.chdir("..")

    # 1) Fresh load — confirm the embedded pipeline structure.
    pipeline = joblib.load(MODEL_PATH)
    print("Loaded artifact:", MODEL_PATH)
    print("Pipeline steps :", [name for name, _ in pipeline.steps])
    assert len(pipeline.steps) == 2, "Expected a 2-step pipeline (preprocessor + model)"
    assert pipeline.steps[0][0] == "preprocessor", "First step must be the preprocessor"
    print("OK — preprocessor is embedded in the artifact.\n")

    # 2) Predict on the sample patients via the production wrapper.
    predictor = HeartDiseasePredictor(MODEL_PATH)

    header = f"| {'Patient':<10} | {'Prediction':>10} | {'Probability':>11} | {'Risk':>9} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    print(header)
    print(sep)
    for patient in SAMPLE_PATIENTS:
        result = predictor.predict(patient["features"])
        # 3) Sanity checks on the output ranges.
        assert result["prediction"] in (0, 1), "prediction must be 0 or 1"
        assert 0.0 <= result["probability"] <= 1.0, "probability must be in [0, 1]"
        assert result["risk_level"] in ("Low", "Moderate", "High")
        print(
            f"| {patient['name']:<10} | {result['prediction']:>10} | "
            f"{result['probability']:>11.4f} | {result['risk_level']:>9} |"
        )

    print("\nAll inference checks passed.")


if __name__ == "__main__":
    main()
