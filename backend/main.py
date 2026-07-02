import os
import logging
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.environ.get("MODEL_NAME", "BatteryHealthModel")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
FEATURE_COUNT = 9

model = None
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except MlflowException as e:
        logging.warning(
            "Could not load model '%s' at stage '%s': %s. "
            "App will start without a model loaded (expected before the "
            "first promotion, or if MLFLOW_TRACKING_URI is stale locally).",
            MODEL_NAME, MODEL_STAGE, e,
        )


def validate_features(data: dict) -> list:
    if "features" not in data:
        raise ValueError("Missing 'features' key")
    features = data["features"]
    if not isinstance(features, list) or len(features) != FEATURE_COUNT:
        raise ValueError(f"'features' must be a list of {FEATURE_COUNT} numbers")
    return [float(v) for v in features]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = validate_features(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    result = model.predict([features])
    return {"prediction": result.tolist() if hasattr(result, "tolist") else list(result)}
