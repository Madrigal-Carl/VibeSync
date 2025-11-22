import pandas as pd
import os

# 1. Load raw Spotify dataset
raw_csv = "backend/data/raw/spotify_songs.csv"
df = pd.read_csv(raw_csv)

# 2. Drop rows with missing essential features
feature_columns = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
df = df.dropna(subset=feature_columns)

# 3. Define the mood function
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

# 4. Apply mood
df["mood"] = df.apply(lambda row: determine_mood(row["valence"], row["energy"]), axis=1)

# 5. Save new CSV
os.makedirs("data/processed", exist_ok=True)
df.to_csv("backend/data/processed/spotify_songs_with_mood.csv", index=False)

print("spotify_songs_with_mood.csv created successfully!")
print(df[["track_name", "artists", "mood"]].head())
