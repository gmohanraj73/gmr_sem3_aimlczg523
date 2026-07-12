"""Integration tests for the FastAPI app in src/api.py (via TestClient)."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from src.api import app

MODEL_PATH = "models/best_model_pipeline.pkl"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="trained model artifact not found — run `python -m src.train` first",
)

client = TestClient(app)

VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 3, "ca": 0, "thal": 6,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_metrics_endpoint_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_request" in response.text


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["model_version"] == "1.0.0"
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_field_returns_422():
    response = client.post("/predict", json={"age": 55})
    assert response.status_code == 422


def test_predict_invalid_type_returns_422():
    payload = {**VALID_PATIENT, "age": "not-a-number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_out_of_range_returns_422():
    payload = {**VALID_PATIENT, "age": 999}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
