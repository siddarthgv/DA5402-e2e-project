import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def preprocess(input_path, output_path):
    logging.info("Loading data...")
    df = pd.read_csv(input_path, na_values=['?'], low_memory=False)

    logging.info(f"Initial shape: {df.shape}")

    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        errors='coerce',
        dayfirst=True
    )

    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    df = df.drop(columns=['Date', 'Time'])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.ffill().dropna()

    logging.info(f"Final shape after cleaning: {df.shape}")

    df.to_csv(output_path)

    logging.info(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    preprocess("data/processed/raw_loaded.csv",
               "data/processed/clean.csv")