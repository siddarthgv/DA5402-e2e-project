import pandas as pd
import numpy as np
import joblib

# LOAD DATA
df = pd.read_csv(
    "data/raw/household_power_consumption.txt",
    sep=';',
    low_memory=False
)

print("Initial shape:", df.shape)

# COMBINE DATE + TIME
df['datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    errors='coerce',
    dayfirst=True
)

# DROP rows where datetime failed
df = df.dropna(subset=['datetime'])

# SET INDEX
df = df.set_index('datetime')

# DROP original Date & Time columns (IMPORTANT)
df = df.drop(columns=['Date', 'Time'])

# HANDLE "?"
df = df.replace('?', pd.NA)

# CONVERT ONLY NUMERIC COLUMNS
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# FILL MISSING VALUES
df = df.ffill()

# DROP remaining NaNs (should be minimal now)
df = df.dropna()

# CHECKS
print("Final shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nData types:\n", df.dtypes)
print("\nSample:\n", df.head())

# =========================
# RESAMPLING (Minute → Hourly)
# =========================
df_hourly = df.resample('h').mean()

print("\nHourly shape:", df_hourly.shape)
print("\nSample hourly data:\n", df_hourly.head())

# =========================
# FEATURE ENGINEERING
# =========================

df_feat = df_hourly.copy()

# -------- Time features --------
df_feat['hour'] = df_feat.index.hour
df_feat['day'] = df_feat.index.day
df_feat['month'] = df_feat.index.month
df_feat['weekday'] = df_feat.index.weekday

# -------- Lag features (VERY IMPORTANT) --------
df_feat['lag_1'] = df_feat['Global_active_power'].shift(1)
df_feat['lag_24'] = df_feat['Global_active_power'].shift(24)

# -------- Rolling statistics --------
df_feat['rolling_mean_24'] = df_feat['Global_active_power'].rolling(24).mean()
df_feat['rolling_std_24'] = df_feat['Global_active_power'].rolling(24).std()

# -------- Drop NaNs created by lag/rolling --------
df_feat = df_feat.dropna()

# =========================
# CHECKS
# =========================
print("\nFeature dataset shape:", df_feat.shape)
print("\nColumns:\n", df_feat.columns)
print("\nSample:\n", df_feat.head())

# =========================
# TRAIN / TEST SPLIT (correct)
# =========================

split_idx = int(len(df_feat) * 0.8)

train = df_feat.iloc[:split_idx]
test = df_feat.iloc[split_idx:]

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)


# =========================
# SELECT FEATURES
# =========================

features = [
    'hour', 'day', 'month', 'weekday',
    'lag_1', 'lag_24',
    'rolling_mean_24', 'rolling_std_24'
]

target = 'Global_active_power'

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# =========================
# BASELINE MODEL (Linear Regression)
# =========================
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

preds_lr = model_lr.predict(X_test)


# =========================
# EVALUATION
# =========================
from sklearn.metrics import mean_squared_error


rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))
print("\nLinear Regression RMSE:", rmse_lr)

# =========================
# XGBOOST MODEL
# =========================

from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_xgb.fit(X_train, y_train)

preds_xgb = model_xgb.predict(X_test)


# =========================
# EVALUATION
# =========================
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
print("\nXGBoost RMSE:", rmse_xgb)


# =========================
# ANOMALY DETECTION (Residual-based)
# =========================

# Residuals = actual - predicted
residuals = y_test - preds_xgb

# Threshold (tunable)
threshold = 2 * residuals.std()

# Detect anomalies
anomalies = abs(residuals) > threshold

# Add to dataframe
test = test.copy()
test['predictions'] = preds_xgb
test['residual'] = residuals
test['anomaly'] = anomalies

print("Residual std:", residuals.std())
print("Threshold:", threshold)
print("\nNumber of anomalies detected:", anomalies.sum())


# Model saving
joblib.dump(model_xgb, "model.pkl")
print("Model saved!")