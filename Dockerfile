FROM apache/airflow:2.8.1-python3.11

USER root

# system deps (keep this stable; rarely change it)
RUN apt-get update && apt-get install -y \
    build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

USER airflow

# IMPORTANT: copy only requirements first (good cache layer)
COPY airflow-requirements.txt /tmp/requirements.txt

# pip cache optimization (works only with BuildKit)
RUN --mount=type=cache,target=/home/airflow/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements.txt