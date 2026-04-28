from fastapi import FastAPI, Response, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import os
from scipy.stats import ks_2samp
import pandas as pd

app = FastAPI(title="Energy Prediction API", version="1.0.0")

REQUEST_COUNT = Counter("api_requests_total", "Total API requests")
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)
PREDICTION_THROUGHPUT = Counter("api_predictions_total", "Total predictions served")
IN_PROGRESS = Gauge("api_requests_in_progress", "Requests currently in progress")
MODEL_LOAD_SUCCESS = Gauge("model_loaded", "Whether model is loaded successfully (1=yes, 0=no)")

PREDICTION_ERRORS = Counter("prediction_errors_total", "Prediction errors logged")
ERROR_RATE = Gauge("model_error_rate", "Model error rate")
DRIFT_SCORE = Gauge("data_drift_score", "Data drift score")

MODEL_PATH= "/app/models/model.pkl"
LOG_PATH = "logs/predictions.csv"

model = None


def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOAD_SUCCESS.set(1)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        MODEL_LOAD_SUCCESS.set(0)
        print(f"Failed to load model: {e}")


load_model()


def log_prediction(features, prediction, actual=None, error=None):
    os.makedirs("logs", exist_ok=True)
    row = [*features, prediction, actual, error]
    df = pd.DataFrame([row])
    file_exists = os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)


def compute_drift():
    if not os.path.exists(LOG_PATH):
        return 0.0

    df = pd.read_csv(LOG_PATH)

    if len(df) < 50:
        return 0.0

    df = df.dropna()

    split = len(df) // 2
    past = df.iloc[:split]
    recent = df.iloc[split:]

    scores = []

    for col in past.columns:
        try:
            stat, _ = ks_2samp(past[col], recent[col])
            scores.append(stat)
        except:
            continue

    return float(sum(scores) / max(len(scores), 1))


class InputData(BaseModel):
    features: list[float]
    actual: float | None = None


@app.get("/")
def home():
    return {"message": "Energy Prediction API running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "model_path": MODEL_PATH}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/reload")
def reload_model():
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model reload failed")
    return {"status": "reloaded", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.inc()
    PREDICTION_THROUGHPUT.inc()
    IN_PROGRESS.inc()

    start = time.time()

    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]

        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)

        error = None

        if data.actual is not None:
            error = data.actual - prediction
            PREDICTION_ERRORS.inc()

        log_prediction(data.features, prediction, data.actual, error)

        try:
            if os.path.exists(LOG_PATH):
                df = pd.read_csv(LOG_PATH)
                if "error" in df.columns:
                    ERROR_RATE.set((df["error"].abs() > df["error"].std()).mean())
        except:
            pass

        drift = compute_drift()
        DRIFT_SCORE.set(drift)

        return {
            "prediction": float(prediction),
            "latency_ms": round(latency * 1000, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        IN_PROGRESS.dec()