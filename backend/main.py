import os
import logging
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, HTTPException

app = FastAPI()

MODEL_NAME = os.environ.get("MODEL_NAME", "BatteryHealthModel")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = model.predict([data["features"]])
    return {"prediction": result.tolist() if hasattr(result, "tolist") else list(result)}