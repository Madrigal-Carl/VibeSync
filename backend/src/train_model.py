import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib

# Load data
X_train = pd.read_csv("backend/data/processed/X_train.csv").values
X_test = pd.read_csv("backend/data/processed/X_test.csv").values
y_train = pd.read_csv("backend/data/processed/y_train.csv")["mood"].values
y_test = pd.read_csv("backend/data/processed/y_test.csv")["mood"].values

le = joblib.load("backend/models/label_encoder.pkl")
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train_enc = torch.tensor(y_train_enc, dtype=torch.long)
y_test_enc = torch.tensor(y_test_enc, dtype=torch.long)

# Neural network with Dropout
class MoodNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MoodNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


input_size = X_train.shape[1]
hidden_size = 64
output_size = len(le.classes_)

model = MoodNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
batch_size = 128

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train_enc[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Evaluate
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_enc).sum().item() / y_test_enc.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "backend/models/mood_nn_model.pth")
print("Model saved with dropout and label smoothing.")
