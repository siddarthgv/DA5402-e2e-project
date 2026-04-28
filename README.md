Welcome to this repository!

MLOps End-to-End Energy Prediction System
========================================

Overview
--------
This project implements an end-to-end MLOps pipeline for energy consumption prediction.
It includes data ingestion, feature engineering, model training, API serving, monitoring,
and automated retraining.

The system is fully containerized using Docker and orchestrated with Airflow.

Project Structure
-----------------
.
├── airflow/                 # Airflow setup and configs
├── dags/
│   └── pipeline_dag.py      # Main Airflow DAG
├── data/
│   ├── raw/                 # Raw input data
│   ├── processed/           # Cleaned datasets
│   └── chunks_head/         # Intermediate splits
├── logs/                    # Prediction logs
├── metrics/                 # Metrics storage
├── mlflow_db/               # MLflow backend database
├── mlruns/                  # MLflow experiment artifacts
├── models/
│   └── model.pkl            # Trained model
├── src/
│   ├── api.py               # FastAPI inference service
│   ├── app.py               # Streamlit frontend
│   ├── ingestion.py         # Data ingestion
│   ├── preprocessing.py     # Data preprocessing
│   ├── feature_engineering.py
│   ├── train.py             # Model training
│   └── evaluate.py          # Model evaluation
├── docker-compose.yaml      # Multi-service orchestration
├── Dockerfile               # Base container
├── Dockerfile.fastapi       # FastAPI service container
├── prometheus.yml           # Prometheus config
├── alerts.yml               # Alert rules
├── airflow.db               # Airflow metadata DB
├── requirements.txt
├── fastapi-requirements.txt
├── airflow-requirements.txt
├── params.yaml              # Model/config parameters
├── dvc.yaml / dvc.lock      # DVC pipeline tracking
├── test.csv                 # Test dataset
└── README.txt

System Components
-----------------
- Streamlit: User interface for predictions and CSV uploads
- FastAPI: Model serving API
- Airflow: Pipeline orchestration
- MLflow: Experiment tracking and model registry
- Prometheus: Metrics collection
- Grafana: Monitoring dashboards
- Docker Compose: Service orchestration

API Endpoints
-------------
POST /predict   -> Returns prediction for input features
GET  /health    -> Service health check
GET  /ready     -> Model readiness check
GET  /metrics   -> Prometheus metrics

Testing
-------

1. CSV-based Testing
-------------------
File: test.csv

Used for batch evaluation and drift detection.
Contains feature columns and (optionally) ground truth values.

Purpose:
- Validate model predictions
- Compute error percentage
- Monitor drift over time

2. Load Testing using CURL
-------------------------
Used to simulate multiple API requests:

for i in {1..20}; do
  curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1,2,3,4,5,6,7,8]}'
done

Purpose:
- Test API responsiveness
- Validate concurrent handling
- Generate metrics for Prometheus

Monitoring
----------
Metrics tracked:
- Total predictions
- Requests in progress
- Model load status
- Prediction errors
- Error rate
- Data drift score

Prometheus collects metrics from FastAPI.
Grafana dashboards visualize:
- API performance
- Model error rate
- Drift trends
- Model load status

Alerts
------
Defined in alerts.yml

Triggers alerts for:
- High error rate
- Model not loaded
- Drift threshold exceeded

Pipeline Flow
-------------
1. Data ingestion
2. Preprocessing
3. Feature engineering
4. Model training
5. Evaluation
6. MLflow logging
7. Deployment via FastAPI
8. Monitoring via Prometheus + Grafana

Run Instructions
----------------
Start all services:

docker-compose up --build

Access:
- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501
- Airflow: http://localhost:8081
- MLflow: http://localhost:5001
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

Notes
-----
- Model is loaded from /app/models/model.pkl
- Logs stored in logs/predictions.csv
- MLflow tracks all experiments and artifacts
- Airflow DAG automates retraining
