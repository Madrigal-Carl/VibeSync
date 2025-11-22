import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

# Directories
os.makedirs("backend/data/processed", exist_ok=True)
os.makedirs("backend/models", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("backend/data/raw/spotify_songs.csv")
print(f"Original dataset shape: {df.shape}")

# 2. Drop unnecessary columns
columns_to_drop = [
    "Unnamed: 0", "track_id", "artists", "album_name", "track_name",
    "popularity", "duration_ms", "explicit", "key", "mode", "time_signature", "track_genre"
]
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Columns after drop: {df.columns.tolist()}")

# 3. Drop missing values
df = df.dropna()
print(f"Shape after dropping missing values: {df.shape}")

# 4. Assign mood
def determine_mood(valence, energy, tempo):
    if valence >= 0.7 and energy >= 0.7:
        return "party" if tempo >= 120 else "happy"
    elif valence >= 0.7 and energy < 0.7:
        return "cheerful" if tempo >= 100 else "relaxed"
    elif 0.5 <= valence < 0.7 and energy >= 0.6:
        return "upbeat" if tempo >= 110 else "energetic"
    elif 0.5 <= valence < 0.7 and 0.4 <= energy < 0.6:
        return "focused" if tempo >= 100 else "calm"
    elif 0.5 <= valence < 0.7 and energy < 0.4:
        return "relaxed" if tempo >= 90 else "mellow"
    elif valence < 0.5 and energy >= 0.6:
        return "intense" if tempo >= 110 else "aggressive"
    elif valence < 0.5 and 0.4 <= energy < 0.6:
        return "sad" if tempo >= 80 else "melancholic"
    elif valence < 0.5 and energy < 0.4:
        return "sad" if tempo >= 70 else "melancholic"
    else:
        return "calm"

df["mood"] = df.apply(lambda row: determine_mood(row["valence"], row["energy"], row["tempo"]), axis=1)
print(f"Mood distribution:\n{df['mood'].value_counts()}")

# 5. Separate features and labels
X = df.drop(columns=["mood"])
y = df["mood"]

# 6. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature order
joblib.dump(scaler, "backend/models/scaler.pkl")
joblib.dump(list(X.columns), "backend/models/feature_columns.pkl")

# 7. Encode labels for neural network
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, "backend/models/label_encoder.pkl")

print("Label encoder classes:", le.classes_)

# 8. Split train/test (BEFORE oversampling)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Convert encoded labels back to original strings for saving
y_train = le.inverse_transform(y_train_enc)
y_test = le.inverse_transform(y_test_enc)

# 9. NOW oversample ONLY on the training set
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train_enc)

print("Training class distribution AFTER oversampling:\n",
      pd.Series(y_train_resampled).value_counts())

print("Testing class distribution (should be unchanged):\n",
      pd.Series(y_test_enc).value_counts())

# 10. Save processed data
pd.DataFrame(X_train_resampled, columns=X.columns).to_csv("backend/data/processed/X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("backend/data/processed/X_test.csv", index=False)
pd.DataFrame(le.inverse_transform(y_train_resampled), columns=["mood"]).to_csv("backend/data/processed/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["mood"]).to_csv("backend/data/processed/y_test.csv", index=False)

print("Preprocessing completed. Scaler, feature columns, and label encoder saved.")
