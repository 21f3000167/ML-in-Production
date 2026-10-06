# ML in Production API

A simple machine learning API built using FastAPI and a trained scikit-learn text classification model.

## What the Model Predicts

The trained model takes a text input and predicts one of two classes:

- `0`
- `1`

The model is loaded from the `model.pkl` file using `joblib`.

## API Endpoints

### GET `/health`

Checks whether the API is running and whether the trained model was loaded successfully.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}