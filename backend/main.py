import os
import time
import logging
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

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

# --- Prometheus : métriques génériques (volume + latence + erreurs HTTP, toutes routes) ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# --- Métriques custom, spécifiques au métier "prédiction" (exigées par le sujet) ---
PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Nombre total de requêtes de prédiction reçues",
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Temps de réponse de l'inférence du modèle sur /predict, en secondes",
)
PREDICTION_FAILURES_TOTAL = Counter(
    "prediction_failures_total",
    "Nombre de requêtes de prédiction ayant échoué",
    ["reason"],
)
APP_UP = Gauge(
    "app_up",
    "1 si le backend est en vie et répond, 0 sinon",
)
APP_START_TIME = time.time()
APP_UPTIME_SECONDS = Gauge(
    "app_uptime_seconds",
    "Durée depuis le démarrage du backend, en secondes",
)

APP_UP.set(1)


def validate_features(data: dict) -> list:
    if "features" not in data:
        raise ValueError("Missing 'features' key")
    features = data["features"]
    if not isinstance(features, list) or len(features) != FEATURE_COUNT:
        raise ValueError(f"'features' must be a list of {FEATURE_COUNT} numbers")
    return [float(v) for v in features]


@app.get("/health")
def health():
    APP_UPTIME_SECONDS.set(time.time() - APP_START_TIME)
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    PREDICTION_REQUESTS_TOTAL.inc()

    if model is None:
        PREDICTION_FAILURES_TOTAL.labels(reason="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = validate_features(data)
    except ValueError as e:
        PREDICTION_FAILURES_TOTAL.labels(reason="invalid_input").inc()
        raise HTTPException(status_code=422, detail=str(e))

    try:
        with PREDICTION_LATENCY_SECONDS.time():
            result = model.predict([features])
    except Exception as e:
        PREDICTION_FAILURES_TOTAL.labels(reason="inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"prediction": result.tolist() if hasattr(result, "tolist") else list(result)}

