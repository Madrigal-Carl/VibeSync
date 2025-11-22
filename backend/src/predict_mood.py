import pandas as pd
import joblib

# 1. Load trained model, scaler, and feature columns
clf = joblib.load("models/mood_classifier_rf.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
print("Model, scaler, and feature columns loaded.")

# 2. Example: New song features (replace with real values)
new_song_features = pd.DataFrame([{
    "danceability": 0.8,
    "energy": 0.7,
    "loudness": -5.0,
    "speechiness": 0.05,
    "acousticness": 0.1,
    "instrumentalness": 0.0,
    "liveness": 0.15,
    "valence": 0.9,
    "tempo": 120.0
}])

# 3. Reorder columns to match training data
new_song_features = new_song_features[feature_columns]

# 4. Scale features
new_song_scaled = scaler.transform(new_song_features)
new_song_scaled = pd.DataFrame(new_song_scaled, columns=feature_columns)  # keep feature names

# 5. Predict mood
predicted_mood = clf.predict(new_song_scaled)
print(f"Predicted mood: {predicted_mood[0]}")
