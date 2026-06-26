# Fashion-MNIST Classification Project

## Project Information

**Course:** EE-439 — Introduction to Machine Learning  
**Project Title:** *Fashion-MNIST Image Classification using Logistic Regression and Convolutional Neural Networks*

### Team Members
- **Muhammad Talha Ayyaz** (2023-EE-174)
- **Abdullah Chaudhary** (2023-EE-168)

---

# Project Overview

This project explores multiple machine learning approaches for image classification on the Fashion-MNIST dataset.

The project includes:

- Logistic Regression from Scratch (NumPy)
- Convolutional Neural Network from Scratch (NumPy)
- Deep CNN using PyTorch

The objective is to compare:
- Linear vs nonlinear models
- Global pixel learning vs spatial feature learning
- Scratch implementations vs framework-based implementations

---

# Required Libraries

Install the required dependencies before running the project:

```bash
pip install numpy pandas torch
```

## Required Python Libraries
- `numpy`
- `pandas`
- `torch` (PyTorch)

---

# Project Directory Structure

```text
.
├── dataset
│   ├── fashion-mnist_test.csv
│   └── fashion-mnist_train.csv
│
├── history
│   ├── cnn_pytorch.csv
│   ├── cnn_scratch_batch_loss_log.csv
│   ├── cnn_scratch_epoch_history.csv
│   └── lr_training_history.csv
│
├── Models
│   ├── cnn_pytorch_weights.pth
│   ├── cnn_scratch_weights.npz
│   ├── cnn_weights_safe_copy.npz
│   └── lr_scratch_weights.npz
│
├── README.md
│
└── src
    ├── CNN_PyTorch.py
    ├── CNN_Scratch.py
    └── LR_Scratch.py
```

---

# Directory Explanation

## `dataset/`
Contains the Fashion-MNIST training and testing datasets in CSV format.

- `fashion-mnist_train.csv` → Training dataset
- `fashion-mnist_test.csv` → Testing dataset

---

## `history/`
Stores training logs and performance metrics.

Examples:
- Training loss
- Validation accuracy
- Batch-wise loss history
- Epoch-wise metrics

---

## `Models/`
Contains saved model weights after training.

| File | Description |
|---|---|
| `cnn_pytorch_weights.pth` | PyTorch CNN weights |
| `cnn_scratch_weights.npz` | Scratch CNN weights |
| `lr_scratch_weights.npz` | Logistic Regression weights |
| `cnn_weights_safe_copy.npz` | Backup copy of CNN weights |

---

## `src/`
Contains all source code implementations.

| Script | Description |
|---|---|
| `LR_Scratch.py` | Logistic Regression from scratch using NumPy |
| `CNN_Scratch.py` | CNN from scratch using NumPy |
| `CNN_PyTorch.py` | Deep CNN implementation using PyTorch |

---

# Models Implemented

## 1. Logistic Regression (Baseline)

- Implemented completely using NumPy
- Uses flattened 784-dimensional image vectors
- Serves as the linear baseline model

---

## 2. CNN from Scratch (Main Model)

Architecture:

```text
Conv → ReLU → MaxPool
      ↓
Conv → ReLU → MaxPool
      ↓
Fully Connected → ReLU
      ↓
Softmax Output
```

Features:
- Custom forward and backward propagation
- `im2col` optimization for fast convolution
- Vectorized max-pooling
- SGD training

---

## 3. Deep CNN (PyTorch)

Features:
- Multiple convolutional layers
- Batch Normalization
- Dropout Regularization
- Adam Optimizer
- Faster convergence and higher accuracy

---

# Dataset

Dataset used:
**Fashion-MNIST**

Classes:
- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

Image size:

```text
28 × 28 grayscale
```

---

# How to Run

Run all commands from the project root directory.

> On Windows, use `python` instead of `python3` if required.

---

## Run Logistic Regression

```bash
python3 src/LR_Scratch.py
```

---

## Run CNN from Scratch

```bash
python3 src/CNN_Scratch.py
```

---

## Run Deep CNN (PyTorch)

```bash
python3 src/CNN_PyTorch.py
```

---

# Training Outputs

During training:
- Training history is saved in `history/`
- Model weights are saved in `Models/`

This allows:
- Reloading trained models
- Plotting learning curves
- Comparing model performance

---

# Final Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 83.64% |
| CNN from Scratch | 90.18% |
| Deep CNN (PyTorch) | 92.99% |

---

# Key Findings

- Linear models struggle with visually similar classes such as:
  - Shirt
  - Coat
  - Pullover
  - T-shirt

- CNNs perform significantly better because they:
  - Learn local spatial features
  - Detect edges and textures
  - Preserve image structure

- The deep CNN achieved the best performance due to:
  - Greater feature extraction capacity
  - Batch normalization
  - Dropout regularization

---

# Technologies Used

- Python
- NumPy
- Pandas
- PyTorch

---

# Authors

## Muhammad Talha Ayyaz
- CNN from Scratch implementation
- EDA and visualization
- Deep CNN implementation
- Model optimization

## Abdullah Chaudhary
- Logistic Regression implementation
- CNN theoretical analysis
- Training evaluation
- Confusion matrix analysis

---

# License

This project is for educational and academic purposes only.