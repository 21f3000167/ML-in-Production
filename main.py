from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="ML Prediction API",
    description="FastAPI service for the trained text-classification model.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    model_error = str(e)
else:
    model_error = None


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictionResponse(BaseModel):
    prediction: int
    probability_class_0: float | None = None
    probability_class_1: float | None = None


@app.get("/")
def root():
    return {"message": "ML API is running. See /docs for usage."}


@app.get("/health")
def health():
    response = {"status": "ok", "model_loaded": model is not None}
    if model_error:
        response["error"] = model_error
    return response


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model failed to load.")

    try:
        prediction = int(model.predict([request.text])[0])

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([request.text])[0]

        return PredictionResponse(
            prediction=prediction,
            probability_class_0=float(probabilities[0]) if probabilities is not None else None,
            probability_class_1=float(probabilities[1]) if probabilities is not None else None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
