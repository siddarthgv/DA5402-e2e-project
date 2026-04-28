import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Energy Consumption MLOps Dashboard")

API_URL = "http://127.0.0.1:8000/predict"
RETRAIN_URL = "http://127.0.0.1:8000/retrain"

# =========================
# MODE SELECTION
# =========================
mode = st.radio("Choose Input Mode", ["Manual Input", "CSV Upload"])

# =========================
# MANUAL INPUT MODE
# =========================
if mode == "Manual Input":
    st.subheader("Enter Feature Values")

    hour = st.number_input("Hour", 0, 23, 12)
    day = st.number_input("Day", 1, 31, 1)
    month = st.number_input("Month", 1, 12, 1)
    weekday = st.number_input("Weekday", 0, 6, 0)

    lag_1 = st.number_input("Lag-1", value=0.0)
    lag_24 = st.number_input("Lag-24", value=0.0)
    rolling_mean_24 = st.number_input("Rolling Mean 24", value=0.0)
    rolling_std_24 = st.number_input("Rolling Std 24", value=0.0)

    if st.button("Predict"):
        features = [hour, day, month, weekday,
                    lag_1, lag_24,
                    rolling_mean_24, rolling_std_24]

        response = requests.post(API_URL, json={"features": features})
        pred = response.json()["prediction"]

        st.success(f"Predicted Energy Consumption: {pred}")

# =========================
# CSV MODE
# =========================
else:
    st.subheader("Upload CSV for Batch Prediction + Monitoring")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview", df.head())

        feature_cols = [
            'hour', 'day', 'month', 'weekday',
            'lag_1', 'lag_24',
            'rolling_mean_24', 'rolling_std_24'
        ]

        X = df[feature_cols].values

        preds = []
        for row in X:
            r = requests.post(API_URL, json={"features": list(row)})
            preds.append(r.json()["prediction"])

        df["prediction"] = preds

        # =========================
        # FEEDBACK LOOP (GROUND TRUTH)
        # =========================
        if "actual" in df.columns:
            df["error"] = df["actual"] - df["prediction"]
            df["abs_error"] = df["error"].abs()

            error_rate = (df["abs_error"] > df["abs_error"].mean()).mean()

            st.metric("Error Rate", f"{error_rate*100:.2f}%")

            # =========================
            # ANOMALY DETECTION
            # =========================
            threshold = 2 * df["error"].std()
            df["anomaly"] = df["abs_error"] > threshold

            anomaly_count = df["anomaly"].sum()
            st.write("Anomalies detected:", anomaly_count)

            # =========================
            # AUTO-RETRAIN TRIGGER
            # =========================
            if anomaly_count > 10:
                st.warning("Anomaly threshold exceeded → Triggering retraining")

                r = requests.post(RETRAIN_URL)
                st.success(r.json()["message"])

            # =========================
            # VISUALIZATION
            # =========================
            fig, ax = plt.subplots()
            ax.plot(df["actual"], label="Actual")
            ax.plot(df["prediction"], label="Prediction")

            anomalies = df[df["anomaly"]]
            ax.scatter(anomalies.index, anomalies["actual"], color="red", label="Anomaly")

            ax.legend()
            st.pyplot(fig)

        else:
            st.line_chart(df["prediction"])