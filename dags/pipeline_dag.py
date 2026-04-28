from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

# ------------------------
# DEFAULT CONFIG
# ------------------------
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ------------------------
# TASK FUNCTIONS
# ------------------------

def ingestion_task():
    import subprocess
    logging.info("Starting ingestion...")
    subprocess.run(["python", "/opt/airflow/src/ingestion.py"], check=True)

def preprocessing_task():
    import subprocess
    logging.info("Starting preprocessing...")
    subprocess.run(["python", "/opt/airflow/src/preprocessing.py"], check=True)

def feature_engineering_task():
    import subprocess
    logging.info("Starting feature engineering...")
    subprocess.run(["python", "/opt/airflow/src/feature_engineering.py"], check=True)

def training_task():
    import subprocess
    logging.info("Starting training...")
    subprocess.run(["python", "/opt/airflow/src/train.py"], check=True)

def evaluation_task():
    import subprocess
    logging.info("Starting evaluation...")
    subprocess.run(["python", "/opt/airflow/src/evaluate.py"], check=True)

# ------------------------
# DAG DEFINITION
# ------------------------

with DAG(
    dag_id="energy_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    description="End-to-end ML pipeline for energy consumption",
) as dag:

    ingestion = PythonOperator(
        task_id="ingestion",
        python_callable=ingestion_task,
    )

    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocessing_task,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_task,
    )

    training = PythonOperator(
        task_id="training",
        python_callable=training_task,
    )

    evaluation = PythonOperator(
        task_id="evaluation",
        python_callable=evaluation_task,
    )

    # DAG FLOW
    ingestion >> preprocessing >> feature_engineering >> training >> evaluation