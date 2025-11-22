from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model, scaler, feature columns ----
model = joblib.load("models/mood_classifier_rf.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ---- Load songs dataset with mood ----
songs_df = pd.read_csv("data/processed/spotify_songs_with_mood.csv")

# ---- Pydantic data model ----
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

# ---- Predict endpoint ----
@app.post("/predict")
def predict_single(features: SongFeatures):
    # Convert input to DataFrame and reorder columns
    df = pd.DataFrame([features.dict()])
    df = df[feature_columns]
    
    # Scale features
    scaled = scaler.transform(df)

    # Predict mood and confidence
    mood = model.predict(scaled)[0]
    confidence = max(model.predict_proba(scaled)[0]) * 100

    # Recommend 3 random songs for this mood
    mood_songs = songs_df[songs_df["mood"] == mood]
    recommended_songs = []
    if len(mood_songs) > 0:
        recommended_songs = mood_songs.sample(min(3, len(mood_songs)))[["track_name", "artists"]].to_dict(orient="records")

    return {
        "mood": mood,
        "confidence": round(confidence, 2),
        "recommended_songs": recommended_songs
    }
