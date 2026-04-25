import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_features(input_path, output_path):
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    df_hourly = df.resample('h').mean()

    df_feat = df_hourly.copy()

    df_feat['hour'] = df_feat.index.hour
    df_feat['day'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat['weekday'] = df_feat.index.weekday

    df_feat['lag_1'] = df_feat['Global_active_power'].shift(1)
    df_feat['lag_24'] = df_feat['Global_active_power'].shift(24)

    df_feat['rolling_mean_24'] = df_feat['Global_active_power'].rolling(24).mean()
    df_feat['rolling_std_24'] = df_feat['Global_active_power'].rolling(24).std()

    df_feat = df_feat.dropna()

    df_feat.to_csv(output_path)

if __name__ == "__main__":
    create_features("data/processed/clean.csv",
                    "data/processed/features.csv")