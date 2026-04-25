import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate(input_path, model_path, metrics_path):
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

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    with open(metrics_path, "w") as f:
        json.dump({"rmse": rmse}, f)

if __name__ == "__main__":
    evaluate("data/processed/features.csv",
             "models/model.pkl",
             "metrics.json")