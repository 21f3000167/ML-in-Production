"""
FastAPI app for serving the trained text-classification model.
"""

import logging
import os
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Model loading
# --------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception:
    model = None
    logger.exception("Failed to load model from %s", MODEL_PATH)


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="ML Production API",
    description="FastAPI service for the trained text-classification model.",
    version="1.0.0",
)


# --------------------------------------------------
# Request/response schemas
# --------------------------------------------------
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


# --------------------------------------------------
# Request logging / monitoring
# --------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "%s %s -> %s | %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response

    except Exception:
        duration_ms = (time.time() - start_time) * 1000
        logger.exception(
            "%s %s -> ERROR | %.2f ms",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "ML API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    try:
        features = np.array([request.text], dtype=object)
        prediction = int(model.predict(features)[0])

        # LogisticRegression provides predict_proba().
        confidence = float(np.max(model.predict_proba(features)[0]))

        logger.info(
            "Prediction completed | prediction=%s | confidence=%.4f",
            prediction,
            confidence,
        )

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc
