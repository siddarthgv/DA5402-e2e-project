import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("⚡ Energy Consumption Dashboard")

API_URL = "http://127.0.0.1:8000/predict"

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview:", df.head())

    # Expecting same feature columns
    features = df[[
    'hour', 'day', 'month', 'weekday',
    'lag_1', 'lag_24',
    'rolling_mean_24', 'rolling_std_24'
    ]].values

    preds = []

    # Call API row-by-row (simple approach)
    for row in features:
        response = requests.post(API_URL, json={"features": list(row)})
        preds.append(response.json()["prediction"])

    df['prediction'] = preds

    # =========================
    # ANOMALY DETECTION
    # =========================
    if 'actual' in df.columns:
        residuals = df['actual'] - df['prediction']
        threshold = 2 * residuals.std()
        df['anomaly'] = abs(residuals) > threshold

        st.write("Anomalies detected:", df['anomaly'].sum())

        # =========================
        # PLOT
        # =========================
        plt.figure(figsize=(12,6))
        plt.plot(df['actual'], label='Actual')
        plt.plot(df['prediction'], label='Predicted')

        anomaly_points = df[df['anomaly']]
        plt.scatter(anomaly_points.index, anomaly_points['actual'], color='red', label='Anomaly')

        plt.legend()
        plt.title("Energy Consumption with Anomalies")
        st.pyplot(plt)

    else:
        st.line_chart(df['prediction'])