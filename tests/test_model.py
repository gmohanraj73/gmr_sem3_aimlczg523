"""Unit tests for HeartDiseasePredictor in src/predict.py.

These require a trained artifact at models/best_model_pipeline.pkl
(produced by `python -m src.train`).
"""
from __future__ import annotations

import os

import pytest

from src.predict import HeartDiseasePredictor

MODEL_PATH = "models/best_model_pipeline.pkl"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="trained model artifact not found — run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def predictor() -> HeartDiseasePredictor:
    return HeartDiseasePredictor(MODEL_PATH)


@pytest.fixture
def sample_patient() -> dict:
    """A valid raw-encoded patient record."""
    return {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0, "thal": 6,
    }


def test_pipeline_loads(predictor):
    assert predictor.pipeline is not None


def test_pipeline_has_two_steps(predictor):
    assert len(predictor.pipeline.steps) == 2


def test_predict_returns_expected_keys(predictor, sample_patient):
    result = predictor.predict(sample_patient)
    assert set(result.keys()) == {"prediction", "probability", "risk_level"}


def test_probability_in_range(predictor, sample_patient):
    result = predictor.predict(sample_patient)
    assert 0.0 <= result["probability"] <= 1.0


def test_prediction_is_binary(predictor, sample_patient):
    result = predictor.predict(sample_patient)
    assert result["prediction"] in (0, 1)


def test_risk_level_valid(predictor, sample_patient):
    result = predictor.predict(sample_patient)
    assert result["risk_level"] in ("Low", "Moderate", "High")


def test_batch_predict_length(predictor, sample_patient):
    results = predictor.predict_batch([sample_patient, sample_patient])
    assert len(results) == 2
