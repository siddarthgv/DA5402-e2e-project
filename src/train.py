import pandas as pd
import joblib
from xgboost import XGBRegressor
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(input_path, model_path):
    df = pd.read_csv(input_path, index_col=0)

    features = [
        'hour', 'day', 'month', 'weekday',
        'lag_1', 'lag_24',
        'rolling_mean_24', 'rolling_std_24'
    ]

    target = 'Global_active_power'

    split_idx = int(len(df) * 0.8)

    train = df.iloc[:split_idx]

    X_train = train[features]
    y_train = train[target]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_model("data/processed/features.csv",
                "models/model.pkl")