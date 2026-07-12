"""FastAPI application serving the heart-disease risk model.

Endpoints:
    GET  /         — service metadata
    GET  /health   — liveness/readiness probe
    POST /predict  — risk prediction for one patient
    GET  /metrics  — Prometheus metrics (added by the instrumentator)

The model is loaded lazily as a process-wide singleton, so the app works both
under a normal Uvicorn startup and inside pytest's TestClient.

NOTE ON FEATURE ENCODING: the field bounds below follow the *original* UCI
Heart Disease encoding (the one returned by ``ucimlrepo`` id=45 and used to
train the model): cp 1-4, restecg 0-2, slope 1-3, ca 0-3, thal in {3,6,7}.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from src.predict import HeartDiseasePredictor

MODEL_VERSION = "1.0.0"
MODEL_PATH = "models/best_model_pipeline.pkl"

# Structured logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# Lazily-initialised singleton predictor.
_predictor: Optional[HeartDiseasePredictor] = None


def get_predictor() -> HeartDiseasePredictor:
    """Return the process-wide predictor, loading it on first use."""
    global _predictor
    if _predictor is None:
        logger.info("Loading model pipeline from %s", MODEL_PATH)
        _predictor = HeartDiseasePredictor(MODEL_PATH)
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the model at startup so the first request isn't slow."""
    try:
        get_predictor()
        logger.info("Model loaded — API ready.")
    except Exception as exc:  # pragma: no cover - defensive startup logging
        logger.error("Model failed to load at startup: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Risk Predictor API",
    description="MLOps Assignment 01 — BITS Pilani AIMLCZG523",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Prometheus metrics auto-instrumentation (exposes GET /metrics).
Instrumentator().instrument(app).expose(app)


class PatientFeatures(BaseModel):
    """Raw patient features (original UCI encoding)."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 1=male, 0=female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression from exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, description="Major vessels colored by fluoroscopy (0-3)")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia: 3=normal, 6=fixed, 7=reversible")


class PredictionResponse(BaseModel):
    # 'model_version' would otherwise clash with pydantic's protected 'model_' namespace.
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    probability: float
    risk_level: str
    model_version: str
    latency_ms: float


@app.get("/")
def root() -> dict:
    return {"message": "Heart Disease Risk Predictor API", "docs": "/docs"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> dict:
    start = time.time()
    try:
        features = patient.model_dump()
        result = get_predictor().predict(features)
        latency = round((time.time() - start) * 1000, 2)
        logger.info(
            "PREDICTION | input_hash=%s | prediction=%s | probability=%s | latency_ms=%s",
            hash(str(features)),
            result["prediction"],
            result["probability"],
            latency,
        )
        return {**result, "model_version": MODEL_VERSION, "latency_ms": latency}
    except Exception as exc:
        logger.error("PREDICTION_ERROR | %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
