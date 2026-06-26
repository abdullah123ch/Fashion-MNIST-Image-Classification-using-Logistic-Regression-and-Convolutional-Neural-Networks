# =========================================================
# EE 439 - Machine Learning Project - Phase 2
# Fashion-MNIST Classification using Flexible Model (PyTorch)
#
# Muhammad Talha Ayyaz (2023-EE-174) - ALL IMPLEMENTATION
#
# Architecture:
#   Conv1 (32) -> BN -> ReLU -> MaxPool
#   Conv2 (64) -> BN -> ReLU -> MaxPool
#   Conv3 (128)-> BN -> ReLU -> MaxPool -> Dropout(0.3)
#   Dense1 (128 -> 64) -> ReLU -> Dropout(0.4)
#   Output (64 -> 10)
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time

# =========================================================
# E174 - UTILS & DATA PREP
# =========================================================

def stratified_split(df, test_size=0.2, seed=42):
    np.random.seed(seed)
    train_idx, val_idx = [], []
    for label in sorted(df['label'].unique()):
        idx = df[df['label'] == label].index.values.copy()
        np.random.shuffle(idx)
        n = int(len(idx) * test_size)
        val_idx.extend(idx[:n])
        train_idx.extend(idx[n:])
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

print("Loading dataset...")
train_data = pd.read_csv("dataset/fashion-mnist_train.csv")
test_data = pd.read_csv("dataset/fashion-mnist_test.csv")
dev, val = stratified_split(train_data)

def prep_data(df):
    X = df.drop('label', axis=1).values.astype(np.float32) / 255.0
    y = df['label'].values.astype(np.int64)
    X = X.reshape(-1, 1, 28, 28)
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = prep_data(dev)
X_val, y_val = prep_data(val)
X_test, y_test = prep_data(test_data)

# =========================================================
# E174 - MODEL DEFINITION
# =========================================================

class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = DeeperCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

WEIGHTS_FILE = "Models/cnn_pytorch_weights.pth"
if os.path.exists(WEIGHTS_FILE):
    print(f"Loading existing weights from {WEIGHTS_FILE}...")
    model.load_state_dict(torch.load(WEIGHTS_FILE))

# =========================================================
# E174 - TRAINING & EVALUATION
# =========================================================

history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}

def get_accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def train(epochs=15, batch_size=64):
    print(f"\nStarting training for {epochs} epochs...\n")
    os.makedirs("Models", exist_ok=True)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        
        epoch_loss, epoch_acc, batches = 0, 0, 0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            b_x, b_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(b_x)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += get_accuracy(outputs, b_y)
            batches += 1
            
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_acc = get_accuracy(val_outputs, y_val)
            
        t_loss, t_acc = epoch_loss/batches, epoch_acc/batches
        e_time = time.time() - start_time
        
        history["epoch"].append(epoch+1)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(round(e_time, 2))
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {e_time:.1f}s")
        
    torch.save(model.state_dict(), WEIGHTS_FILE)
    pd.DataFrame(history).to_csv("Models/cnn_pytorch_training_history.csv", index=False)
    print(f"\nFinal weights saved to {WEIGHTS_FILE}")
    print("Training history saved to Models/cnn_pytorch_training_history.csv")

def test():
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_acc = get_accuracy(test_outputs, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    choice = input("\n\nEnter 'train' to train the model or 'test' to evaluate on test set\nEnter 'both' to do both: ").strip().lower()
    if choice == "train": train()
    elif choice == "test": test()
    elif choice == "both": train(); test()
    else: print("\nInvalid choice.")