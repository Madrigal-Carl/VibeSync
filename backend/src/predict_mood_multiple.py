import pandas as pd
import joblib

# 1. Load trained model, scaler, and feature columns
clf = joblib.load("models/mood_classifier_rf.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
print("Model, scaler, and feature columns loaded.")

# 2. Define new song features (example songs)
new_songs = pd.DataFrame([
    {"danceability": 0.8, "energy": 0.7, "loudness": -5.0, "speechiness": 0.05, "acousticness": 0.1, "instrumentalness": 0.0, "liveness": 0.15, "valence": 0.9, "tempo": 120.0},  # happy
    {"danceability": 0.4, "energy": 0.3, "loudness": -15.0, "speechiness": 0.04, "acousticness": 0.7, "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.2, "tempo": 80.0},   # sad
    {"danceability": 0.6, "energy": 0.7, "loudness": -6.0, "speechiness": 0.05, "acousticness": 0.2, "instrumentalness": 0.0, "liveness": 0.2, "valence": 0.5, "tempo": 130.0},   # energetic
    {"danceability": 0.5, "energy": 0.3, "loudness": -12.0, "speechiness": 0.04, "acousticness": 0.6, "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.65, "tempo": 90.0},   # calm
    {"danceability": 0.5, "energy": 0.5, "loudness": -10.0, "speechiness": 0.05, "acousticness": 0.3, "instrumentalness": 0.0, "liveness": 0.15, "valence": 0.5, "tempo": 100.0}    # neutral
])

# 3. Reorder columns to match training data
new_songs = new_songs[feature_columns]

# 4. Scale features
new_songs_scaled = scaler.transform(new_songs)
new_songs_scaled = pd.DataFrame(new_songs_scaled, columns=feature_columns)  # keep feature names

# 5. Predict moods
predicted_moods = clf.predict(new_songs_scaled)

# 6. Show results
for i, mood in enumerate(predicted_moods):
    print(f"Song {i+1} predicted mood: {mood}")
