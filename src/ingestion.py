import pandas as pd
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(path):
    df = pd.read_csv(path, sep=';', low_memory=False)
    return df

if __name__ == "__main__":
    base = os.getenv("AIRFLOW_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    input_path = os.getenv("DATA_PATH", os.path.join(base, "data/raw/household_power_consumption.txt"))
    output_dir = os.getenv("PROCESSED_DIR", os.path.join(base, "data/processed"))

    df = load_data(input_path)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "raw_loaded.csv"), index=False)
    logging.info("Ingestion complete. Rows loaded: %d", len(df))