import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load raw songs CSV
df = pd.read_csv("backend/data/raw/spotify_songs.csv")
feature_columns = joblib.load("backend/models/feature_columns.pkl")
scaler = joblib.load("backend/models/scaler.pkl")
le = joblib.load("backend/models/label_encoder.pkl")

# Define neural network architecture
class MoodNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)

# Load trained model
input_size = len(feature_columns)
hidden_size = 64
output_size = len(le.classes_)
model = MoodNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("backend/models/mood_nn_model.pth"))
model.eval()

# Prepare features
X = df[feature_columns]
X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict moods using the model
with torch.no_grad():
    outputs = model(X_tensor)
    probs = F.softmax(outputs, dim=1)
    predicted_idx = torch.argmax(probs, axis=1)
    df["mood"] = le.inverse_transform(predicted_idx.numpy())

# Save CSV for recommendations
df[["track_name", "artists", "mood"]].to_csv("backend/data/processed/spotify_songs_with_mood.csv", index=False)
