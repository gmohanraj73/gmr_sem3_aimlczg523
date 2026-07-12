"""Inference wrapper around the saved best-model pipeline.

The pickled pipeline embeds the fitted preprocessor, so callers pass *raw*
feature dictionaries (exactly the columns present in the raw dataset) and never
touch scaling/encoding themselves.
"""
from __future__ import annotations

from typing import Dict, List

import joblib
import pandas as pd


class HeartDiseasePredictor:
    """Load the trained pipeline once and serve single / batch predictions."""

    def __init__(self, model_path: str = "models/best_model_pipeline.pkl") -> None:
        self.pipeline = joblib.load(model_path)

    def predict(self, features: Dict) -> Dict:
        """Predict heart-disease risk for one raw feature dict.

        Returns a dict with the binary prediction, positive-class probability,
        and a human-readable risk band.
        """
        df = pd.DataFrame([features])
        prob = float(self.pipeline.predict_proba(df)[0][1])
        pred = int(prob >= 0.5)
        risk = "High" if prob >= 0.7 else "Moderate" if prob >= 0.4 else "Low"
        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": risk,
        }

    def predict_batch(self, records: List[Dict]) -> List[Dict]:
        """Predict for a list of raw feature dicts."""
        return [self.predict(r) for r in records]


if __name__ == "__main__":
    # Smoke test with the shared high-risk sample patient.
    sample_patient = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 0, "ca": 0, "thal": 1,
    }
    predictor = HeartDiseasePredictor()
    print("Sample patient prediction:")
    print(predictor.predict(sample_patient))
