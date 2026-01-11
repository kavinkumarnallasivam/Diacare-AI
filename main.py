from fastapi import FastAPI
import joblib
import numpy as np

# API
app = FastAPI()

model = joblib.load("diabetes_risk_model.pkl")

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["avg_glucose"],
        data["carbs_g"],
        data["adherence_pct"],
        data["activity_min"]
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    return {
        "risk": int(pred),
        "confidence": round(float(prob), 2)
    }


