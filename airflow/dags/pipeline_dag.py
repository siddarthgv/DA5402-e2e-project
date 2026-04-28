from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT_DIR = "/Users/gvsiddarth/Downloads/mlops-e2e-project"
PYTHON_PATH = f"{PROJECT_DIR}/venv/bin/python"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    dag_id="energy_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    ingestion = BashOperator(
        task_id="ingestion",
        bash_command=f"{PYTHON_PATH} {PROJECT_DIR}/src/ingestion.py"
    )

    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command=f"{PYTHON_PATH} {PROJECT_DIR}/src/preprocessing.py"
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"{PYTHON_PATH} {PROJECT_DIR}/src/feature_engineering.py"
    )

    training = BashOperator(
        task_id="training",
        bash_command=f"{PYTHON_PATH} {PROJECT_DIR}/src/train.py"
    )

    evaluation = BashOperator(
        task_id="evaluation",
        bash_command=f"{PYTHON_PATH} {PROJECT_DIR}/src/evaluate.py"
    )

    ingestion >> preprocessing >> feature_engineering >> training >> evaluation