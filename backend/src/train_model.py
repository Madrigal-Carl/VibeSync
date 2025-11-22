# src/train_model.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load preprocessed (already scaled) data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# 2. Save the feature columns for prediction consistency
feature_columns = list(X_train.columns)
os.makedirs("models", exist_ok=True)
joblib.dump(feature_columns, "models/feature_columns.pkl")
print("Feature columns saved.")

# 3. Initialize Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

# 4. Train the model
clf.fit(X_train, y_train)
print("Model training completed.")

# 5. Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Save the trained model
joblib.dump(clf, "models/mood_classifier_rf.pkl")
print("Trained model saved to models/mood_classifier_rf.pkl")
