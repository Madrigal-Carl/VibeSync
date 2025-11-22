from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model/scaler/columns once on startup
model = joblib.load("models/mood_classifier_rf.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# ---- DATA MODELS ----

class SongFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

class SongBatch(BaseModel):
    songs: List[SongFeatures]


# ---- SINGLE PREDICTION ----

@app.post("/predict")
def predict_single(features: SongFeatures):
    df = pd.DataFrame([features.dict()])
    df = df[feature_columns]
    scaled = scaler.transform(df)

    mood = model.predict(scaled)[0]
    confidence = max(model.predict_proba(scaled)[0]) * 100

    return {
        "mood": mood,
        "confidence": round(confidence, 2)
    }


# ---- MULTIPLE PREDICTION ----

@app.post("/predict/batch")
def predict_batch(data: SongBatch):
    df = pd.DataFrame([s.dict() for s in data.songs])
    df = df[feature_columns]

    scaled = scaler.transform(df)

    moods = model.predict(scaled)
    probs = model.predict_proba(scaled)

    results = []
    for i, mood in enumerate(moods):
        results.append({
            "mood": mood,
            "confidence": round(max(probs[i]) * 100, 2)
        })

    return {"results": results}
