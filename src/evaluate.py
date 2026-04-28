import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate(input_path, model_path, metrics_path):

    logging.info("Loading dataset...")
    df = pd.read_csv(input_path, index_col=0)

    features = [
        'hour', 'day', 'month', 'weekday',
        'lag_1', 'lag_24',
        'rolling_mean_24', 'rolling_std_24'
    ]

    target = 'Global_active_power'

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:]

    X_test = test[features]
    y_test = test[target]

    logging.info("Loading model...")
    model = joblib.load(model_path)

    logging.info("Running predictions...")
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    logging.info(f"RMSE: {rmse}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump({"rmse": rmse}, f)

    logging.info(f"Metrics saved at {metrics_path}")


if __name__ == "__main__":
    base = os.getenv("AIRFLOW_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    processed_dir = os.getenv("PROCESSED_DIR", os.path.join(base, "data/processed"))
    models_dir = os.getenv("MODELS_DIR", os.path.join(base, "models"))
    metrics_dir = os.getenv("METRICS_DIR", os.path.join(base, "metrics"))

    evaluate(
        input_path=os.path.join(processed_dir, "features.csv"),
        model_path=os.path.join(models_dir, "model.pkl"),
        metrics_path=os.path.join(metrics_dir, "metrics.json")
    )