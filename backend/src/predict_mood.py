import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# -----------------------------
# Load scaler, feature columns, label encoder
# -----------------------------
scaler = joblib.load("backend/models/scaler.pkl")
feature_columns = joblib.load("backend/models/feature_columns.pkl")
le = joblib.load("backend/models/label_encoder.pkl")

# Load dataset with moods (for recommendations)
songs_df = pd.read_csv("backend/data/processed/spotify_songs_with_mood.csv")

# -----------------------------
# Define SAME neural network architecture (with dropout)
# -----------------------------
class MoodNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MoodNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load trained model weights
# -----------------------------
input_size = len(feature_columns)
hidden_size = 64
output_size = len(le.classes_)

model = MoodNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("backend/models/mood_nn_model.pth", map_location=torch.device("cpu")))
model.eval()  # important: disables dropout in prediction

# -----------------------------
# Generate random features
# -----------------------------
features = {
    "danceability": round(random.uniform(0, 1), 2),
    "energy": round(random.uniform(0, 1), 2),
    "loudness": round(random.uniform(-60, 0), 2),
    "speechiness": round(random.uniform(0, 1), 2),
    "acousticness": round(random.uniform(0, 1), 2),
    "instrumentalness": round(random.uniform(0, 1), 2),
    "liveness": round(random.uniform(0, 1), 2),
    "valence": round(random.uniform(0, 1), 2),
    "tempo": round(random.uniform(60, 200), 2),
}

# -----------------------------
# Convert to DataFrame & scale
# -----------------------------
df = pd.DataFrame([features])[feature_columns]
scaled = scaler.transform(df)
tensor_input = torch.tensor(scaled, dtype=torch.float32)

# -----------------------------
# Predict mood
# -----------------------------
with torch.no_grad():
    outputs = model(tensor_input)
    probs = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probs, 1)

mood = le.inverse_transform(predicted_idx.numpy())[0]
confidence = confidence.item()

# -----------------------------
# Recommend songs
# -----------------------------
mood_songs = songs_df[songs_df["mood"] == mood]

recommended_songs = []
if not mood_songs.empty:
    recommended_songs = mood_songs.sample(min(3, len(mood_songs)))[
        ["track_name", "artists"]
    ].to_dict(orient="records")

# -----------------------------
# Output
# -----------------------------
print("Randomized Features:", features)
print("Predicted Mood:", mood)
print("Confidence:", round(confidence * 100, 2), "%")
print("Recommended Songs:", recommended_songs)
