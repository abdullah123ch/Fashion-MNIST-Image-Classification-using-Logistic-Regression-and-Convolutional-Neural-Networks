# =========================================================
# EE 439 - Machine Learning Project - Phase 2
# Fashion-MNIST Classification using Logistic Regression from Scratch
#
# Muhammad Talha Ayyaz  (2023-EE-174)
# Abdullah Chaudhary    (2023-EE-168)
#
# Architecture:
#   Input (784) -> Linear (10) -> Softmax
#
# Training:
#   Epochs      : 300
#   LR          : 0.5
#   Loss        : Cross-Entropy
#   Optimizer   : Full-Batch Gradient Descent
# =========================================================

# =========================================================
# E168 -> 2023-EE-168 (Abdullah Chaudhary)
# - ALL IMPLEMENTATION (Architecture, Forward/Backward, 
#   Utils, Data Pipeline, History Logging & Weight Saving)
# =========================================================

import numpy as np
import pandas as pd
import os
import time

# =========================================================
# E168 - UTILITY FUNCTIONS
# =========================================================

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def one_hot(y, n_classes=10):
    m = y.shape[0]
    y_oh = np.zeros((m, n_classes))
    y_oh[np.arange(m), y] = 1
    return y_oh

def loss(y_oh, probs):
    m = y_oh.shape[0]
    return -np.sum(y_oh * np.log(probs + 1e-10)) / m

def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)

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

# =========================================================
# E168 - LOAD & PREP DATASET
# =========================================================

print("Loading dataset...")
train_data = pd.read_csv("dataset/fashion-mnist_train.csv")
test_data = pd.read_csv("dataset/fashion-mnist_test.csv")

dev, val = stratified_split(train_data)

X_dev = dev.drop('label', axis=1).values / 255.0
y_dev = dev['label'].values

X_val = val.drop('label', axis=1).values / 255.0
y_val = val['label'].values

X_test = test_data.drop('label', axis=1).values / 255.0
y_test = test_data['label'].values

print("Train:", X_dev.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# =========================================================
# E168 - WEIGHTS INITIALIZATION
# =========================================================

WEIGHTS_FILE = "Models/lr_scratch_weights.npz"
HISTORY_FILE = "Models/lr_training_history.csv"

m, n = X_dev.shape
n_classes = 10

if os.path.exists(WEIGHTS_FILE):
    print(f"Loading existing weights from {WEIGHTS_FILE}...")
    weights = np.load(WEIGHTS_FILE)
    W = weights['W']
    b = weights['b']
else:
    print("No existing weights found. Initializing from scratch...")
    W = np.random.randn(n, n_classes) * 0.01
    b = np.zeros((1, n_classes))

# =========================================================
# E168 - TRAINING LOGIC & HISTORY
# =========================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "epoch_time": []
}

def train(epochs=300, lr=0.5):
    global W, b
    os.makedirs("Models", exist_ok=True)
    
    Y_oh = one_hot(y_dev)
    Y_val_oh = one_hot(y_val)
    
    print(f"\nStarting training for {epochs} epochs...\n")
    start_time = time.time()

    for ep in range(epochs):
        ep_start = time.time()
        
        # Forward pass
        z = X_dev @ W + b
        probs = softmax(z)

        # Backward pass
        dz = probs - Y_oh
        dW = (1/m) * X_dev.T @ dz
        db = (1/m) * dz.sum(axis=0, keepdims=True)

        # Update weights
        W -= lr * dW
        b -= lr * db

        # Calculate metrics for history
        l = loss(Y_oh, probs)
        train_accuracy = acc(y_dev, probs.argmax(axis=1))
        
        z_val = X_val @ W + b
        probs_val = softmax(z_val)
        val_l = loss(Y_val_oh, probs_val)
        val_accuracy = acc(y_val, probs_val.argmax(axis=1))
        
        ep_time = time.time() - ep_start

        # Append to history
        history["epoch"].append(ep + 1)
        history["train_loss"].append(l)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_l)
        history["val_acc"].append(val_accuracy)
        history["epoch_time"].append(round(ep_time, 4))

        # Logging
        if ep % 30 == 0 or ep == epochs - 1:
            print(f"Epoch {ep:3d} | Train Loss: {l:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_l:.4f} | Val Acc: {val_accuracy:.4f} | Time: {ep_time:.4f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds.")

    # Save weights and history
    np.savez(WEIGHTS_FILE, W=W, b=b)
    pd.DataFrame(history).to_csv(HISTORY_FILE, index=False)
    print(f"Final weights saved to {WEIGHTS_FILE}")
    print(f"Training history saved to {HISTORY_FILE}")

# =========================================================
# E168 - TESTING
# =========================================================

def test():
    print("\nEvaluating on test set...")
    z_test = X_test @ W + b
    probs_test = softmax(z_test)
    
    test_predicted = np.argmax(probs_test, axis=1)
    test_accuracy = acc(y_test, test_predicted)

    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    choice = input("\n\nEnter 'train' to train the model or 'test' to evaluate on test set\nEnter 'both' to do both: ").strip().lower()
    if choice == "train": train()
    elif choice == "test": test()
    elif choice == "both": train(); test()
    else: print("\nInvalid choice.")