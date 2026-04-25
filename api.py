from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Energy Predictor API running"}



class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}