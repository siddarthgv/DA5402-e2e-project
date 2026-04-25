import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def load_data(path):
    df = pd.read_csv(path, sep=';', low_memory=False)
    return df

if __name__ == "__main__":
    df = load_data("data/raw/household_power_consumption.txt")
    df.to_csv("data/processed/raw_loaded.csv", index=False)