import pandas as pd
import joblib
from xgboost import XGBRegressor
import numpy as np
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import os

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("energy-predictor-gold")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_model(input_path, model_path, metrics_path):
    logging.info("Loading dataset...")
    df = pd.read_csv(input_path, index_col=0)

    features = ['hour', 'day', 'month', 'weekday', 'lag_1', 'lag_24', 'rolling_mean_24', 'rolling_std_24']
    target = 'Global_active_power'

    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # ------------------ Linear Regression ------------------
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
        logging.info(f"Linear Regression RMSE: {rmse_lr}")
        mlflow.log_params({"model": "linear_regression", "train_size": len(train), "test_size": len(test)})
        mlflow.log_metric("rmse", rmse_lr)
        mlflow.sklearn.log_model(lr, "model")

    # ------------------ XGBoost ------------------
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)

    with mlflow.start_run(run_name="XGBoost"):
        model.fit(X_train, y_train)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        logging.info(f"XGBoost RMSE: {rmse_xgb}")

        mlflow.log_params({**model.get_params(), "train_size": len(train), "test_size": len(test)})
        mlflow.log_metric("rmse", rmse_xgb)
        mlflow.log_param("input_path", input_path)
        mlflow.xgboost.log_model(model, "model")

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"rmse": float(rmse_xgb)}, f)

    # Save model for API
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved at {model_path}")


if __name__ == "__main__":
    base = os.getenv("AIRFLOW_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    processed_dir = os.getenv("PROCESSED_DIR", os.path.join(base, "data/processed"))
    models_dir = os.getenv("MODELS_DIR", os.path.join(base, "models"))
    metrics_dir = os.getenv("METRICS_DIR", os.path.join(base, "metrics"))

    train_model(
        input_path=os.path.join(processed_dir, "features.csv"),
        model_path=os.path.join(models_dir, "model.pkl"),
        metrics_path=os.path.join(metrics_dir, "metrics.json")
    )