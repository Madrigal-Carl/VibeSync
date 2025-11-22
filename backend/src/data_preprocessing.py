# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/raw/spotify_songs.csv")
print(f"Original dataset shape: {df.shape}")

# 2. Drop unnecessary columns, including 'Unnamed: 0' if present
columns_to_drop = [
    "Unnamed: 0", "track_id", "artists", "album_name", "track_name", 
    "popularity", "duration_ms", "explicit", "key", "mode", "time_signature", "track_genre"
]
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Columns after drop: {df.columns.tolist()}")

# 3. Handle missing values
df = df.dropna()
print(f"Shape after dropping missing values: {df.shape}")

# 4. Create mood label
def determine_mood(valence, energy):
    if valence >= 0.7 and energy >= 0.6:
        return "happy"
    elif valence >= 0.6 and energy < 0.4:
        return "calm"
    elif valence < 0.4 and energy < 0.5:
        return "sad"
    elif valence < 0.5 and energy >= 0.6:
        return "energetic"
    else:
        return "neutral"

df["mood"] = df.apply(lambda row: determine_mood(row["valence"], row["energy"]), axis=1)
print(f"Original mood distribution:\n{df['mood'].value_counts()}")

# 5. Separate features and labels
X = df.drop(columns=["mood"])
y = df["mood"]

# 6. Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature order
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

# 7. Handle class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
print(f"Mood distribution after oversampling:\n{pd.Series(y_resampled).value_counts()}")

# 8. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# 9. Save processed data
pd.DataFrame(X_train, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing completed. Scaler and feature order saved.")
