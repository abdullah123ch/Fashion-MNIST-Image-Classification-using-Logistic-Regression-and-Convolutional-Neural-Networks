# =========================================================
# EE 439 - Machine Learning Project - Phase 2
# Fashion-MNIST Classification using CNN from Scratch
#
# Muhammad Talha Ayyaz  (2023-EE-174)
# Abdullah Chaudhary    (2023-EE-168)
#
# Architecture:
#   Conv1 (32 filters, 3x3) -> ReLU -> MaxPool
#   Conv2 (64 filters, 3x3) -> ReLU -> MaxPool
#   Dense (1600 -> 128)     -> ReLU
#   Output (128 -> 10)      -> Softmax
#
# Training:
#   Epochs 1-5  : LR = 0.01
#   Epochs 6-17 : LR = 0.005
#   Epochs 18+  : LR = 0.001
#   Batch Size  : 32
#   Loss        : Cross-Entropy
#   Optimizer   : SGD (vanilla)
# =========================================================

# =========================================================
# E174 -> 2023-EE-174 (Muhammad Talha Ayyaz) 
# E168 -> 2023-EE-168 (Abdullah Chaudhary)
# =========================================================


import numpy as np
import pandas as pd
import os  # Added for file checking

# =========================================================
# ACTIVATION FUNCTIONS
# =========================================================

# E174
def relu(x):
    return np.maximum(0, x)

# E174
def relu_backward(d_out, x):
    dx = d_out.copy() 

    dx[x <= 0] = 0 # Gradient is zero where input was non-positive
    return dx

# E168
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True) # for numerical stability (preventing overflow) [Suggested by AI] - 2023-EE-168 (Abdullah Chaudhary)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# E168 - Optimized By AI
def cross_entropy(pred, labels):
    batch_size = labels.shape[0]

    loss = -np.log(
        pred[np.arange(batch_size), labels] + 1e-10 # Adding a small epsilon to prevent log(0) which can cause numerical instability
    )

    return np.mean(loss)


# =========================================================
# IM2COL
# =========================================================

# E174 - Optimized By AI
def im2col(images, kernel_size=3):
    batch_size, channels, height, width = images.shape

    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    cols = []

    for y in range(out_height):
        for x in range(out_width):
            patch = images[
                :,
                :,
                y:y+kernel_size,
                x:x+kernel_size
            ] # shape: (batch_size, channels, kernel_size, kernel_size)

            cols.append(
                patch.reshape(batch_size, -1)
            ) # shape: (batch_size, channels * kernel_size * kernel_size)

    cols = np.stack(cols, axis=1)
    cols = cols.reshape(
        batch_size * out_height * out_width,
        -1
    )
    
    # print("im2col output shape:", cols.shape)
    return cols

# =========================================================
# COL2IM
# =========================================================

# E174 - Optimized By AI
def col2im(cols, image_shape, kernel_size=3):
    batch_size, channels, height, width = image_shape

    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    images = np.zeros(image_shape)

    cols = cols.reshape(
        batch_size,
        out_height * out_width,
        channels * kernel_size * kernel_size
    )

    patch_num = 0
    for y in range(out_height):
        for x in range(out_width):
            patch = cols[:, patch_num, :]

            patch = patch.reshape(
                batch_size,
                channels,
                kernel_size,
                kernel_size
            )

            images[
                :,
                :,
                y:y+kernel_size,
                x:x+kernel_size
            ] += patch

            patch_num += 1
    return images


# =========================================================
# CONVOLUTION FORWARD
# =========================================================

# E168 
def conv_forward(images, filters, biases):

    batch_size, channels, height, width = images.shape

    num_filters = filters.shape[0]

    image_cols = im2col(images) 
    # shape: (batch_size * out_height * out_width, channels * kernel_size * kernel_size)

    filter_cols = filters.reshape(
        num_filters,
        -1
    ) # shape: (num_filters, channels * kernel_size * kernel_size)

    output = image_cols @ filter_cols.T # Y = XW^T + b, where X=image_cols, W=filter_cols

    output += biases

    out_height = height - 3 + 1
    out_width = width - 3 + 1

    output = output.reshape(
        batch_size,
        out_height,
        out_width,
        num_filters
    )

    output = output.transpose(0, 3, 1, 2) 
    
    # print("conv_forward output shape:", output.shape)

    return output, image_cols

# =========================================================
# CONVOLUTION BACKWARD
# =========================================================

# E168
def conv_backward(
    d_out,
    image_cols,
    filters,
    image_shape
):
    batch_size = image_shape[0]

    num_filters = filters.shape[0]

    d_out = d_out.transpose(
        0,
        2,
        3,
        1
    ).reshape(-1, num_filters)

    filter_cols = filters.reshape(
        num_filters,
        -1
    )

    d_filters = d_out.T @ image_cols

    d_filters = d_filters.reshape(
        filters.shape
    )

    d_biases = np.sum(
        d_out,
        axis=0
    )

    d_cols = d_out @ filter_cols

    d_images = col2im(
        d_cols,
        image_shape
    )

    return d_images, d_filters, d_biases


# =========================================================
# MAXPOOL FORWARD
# =========================================================

# E174
# def maxpool_forward(images):

#     batch_size, channels, height, width = images.shape

#     output = np.zeros((
#         batch_size,
#         channels,
#         height // 2,
#         width // 2
#     ))

#     max_positions = {}

#     for b in range(batch_size):

#         for c in range(channels):

#             for i in range(height // 2):

#                 for j in range(width // 2):

#                     region = images[
#                         b,
#                         c,
#                         i*2:(i+1)*2,
#                         j*2:(j+1)*2
#                     ]

#                     max_pos = np.unravel_index(
#                         np.argmax(region),
#                         region.shape
#                     )

#                     max_positions[
#                         (b, c, i, j)
#                     ] = max_pos

#                     output[b, c, i, j] = np.max(region)

#     return output, max_positions

# E174 - Stack Overflow Inspired Optimization
def maxpool_forward(images):
    batch_size, channels, height, width = images.shape
    
    # crop to even dimensions if needed
    h = (height // 2) * 2
    w = (width // 2) * 2
    images = images[:, :, :h, :w]
    
    oh, ow = h // 2, w // 2
    x = images.reshape(batch_size, channels, oh, 2, ow, 2)
    output = x.max(axis=(3, 5))
    max_positions = (x == output[:, :, :, np.newaxis, :, np.newaxis]) # Remember the positions of the max values for backward pass
    
    return output, max_positions


# =========================================================
# MAXPOOL BACKWARD
# =========================================================

# E168
# def maxpool_backward(
#     d_out,
#     max_positions,
#     image_shape
# ):

#     batch_size, channels, height, width = image_shape

#     d_images = np.zeros(image_shape)

#     for b in range(batch_size):

#         for c in range(channels):

#             for i in range(height // 2):

#                 for j in range(width // 2):

#                     x, y = max_positions[
#                         (b, c, i, j)
#                     ]

#                     d_images[
#                         b,
#                         c,
#                         i*2+x,
#                         j*2+y
#                     ] = d_out[b, c, i, j]

#     return d_images

# E174 - Stack Overflow Inspired Optimization 
def maxpool_backward(d_out, max_positions, image_shape):
    batch_size, channels, height, width = image_shape
    h = (height // 2) * 2
    w = (width // 2) * 2
    oh, ow = h // 2, w // 2

    d_expanded = d_out[:, :, :, np.newaxis, :, np.newaxis]
    d_cropped = (max_positions * d_expanded).reshape(batch_size, channels, h, w)
    
    # pad back to original shape with zeros
    d_images = np.zeros(image_shape)
    d_images[:, :, :h, :w] = d_cropped
    
    return d_images

# =========================================================
# STRATIFIED SPLIT
# =========================================================

# E168 - AI Generated Code - Stratified Split Function
def stratified_split(
    df,
    test_size=0.2,
    seed=42
):
    """Splits the DataFrame into stratified train and validation sets."""
    np.random.seed(seed)

    train_idx = []
    val_idx = []

    for label in sorted(df['label'].unique()):

        idx = df[
            df['label'] == label
        ].index.values.copy()

        np.random.shuffle(idx)

        n = int(len(idx) * test_size)

        val_idx.extend(idx[:n])

        train_idx.extend(idx[n:])

    train_df = df.iloc[train_idx]

    val_df = df.iloc[val_idx]

    train_df = train_df.reset_index(drop=True)

    val_df = val_df.reset_index(drop=True)

    return train_df, val_df

# =========================================================================================================
# ABOVE ARE ONLY IMPORTS AND FUNCTION DEFINITIONS - NO EXECUTION HAPPENS UNTIL THIS POINT
# BELOW IS THE MAIN EXECUTION CODE THAT LOADS DATA, TRAINS THE MODEL, AND EVALUATES IT
# =========================================================================================================

# =========================================================
# LOAD DATASET
# =========================================================

train_data = pd.read_csv(
    "dataset/fashion-mnist_train.csv"
)

test_data = pd.read_csv(
    "dataset/fashion-mnist_test.csv"
)

dev, val = stratified_split(train_data)

# =========================================================
# TRAIN DATA
# =========================================================

x_train = dev.drop(
    'label',
    axis=1
).values

y_train = dev['label'].values

x_train = x_train / 255.0 # Normalize pixel values to [0, 1]

x_train = x_train.reshape(
    -1,     # Number of samples (keep the same)
    1,      # Number of channels (1 for grayscale)
    28,     # Height of the image
    28      # Width of the image
)


# =========================================================
# VALIDATION DATA
# =========================================================

x_val = val.drop(
    'label',
    axis=1
).values

y_val = val['label'].values

x_val = x_val / 255.0

x_val = x_val.reshape(
    -1,
    1,
    28,
    28
)


# =========================================================
# TEST DATA
# =========================================================

x_test = test_data.drop(
    'label',
    axis=1
).values

y_test = test_data['label'].values

x_test = x_test / 255.0

x_test = x_test.reshape(
    -1,
    1,
    28,
    28
)


print("Train:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)


# =========================================================
# WEIGHTS
# =========================================================

# Save / Load weights file mechanism generated by AI - 2023-EE-168 (Abdullah Chaudhary)

WEIGHTS_FILE = "Models/cnn_scratch_weights.npz"

if os.path.exists(WEIGHTS_FILE):
    print(f"Loading existing weights from {WEIGHTS_FILE}...")
    weights = np.load(WEIGHTS_FILE)
    
    filters1 = weights['filters1']
    bias1 = weights['bias1']
    
    filters2 = weights['filters2']
    bias2 = weights['bias2']
    
    w3 = weights['w3']
    b3 = weights['b3']
    
    w4 = weights['w4']
    b4 = weights['b4']
else:
    print("No existing weights found. Initializing from scratch...")
    
    """
    np.sqrt(2 / fan_in) is a common initialization strategy known as He initialization,
    which helps maintain stable gradients during training.
    The fan_in is the number of input connections to the layer,
    which for convolutional layers is calculated as (number of input channels * kernel height * kernel width).
    For dense layers, it's simply the number of input features.
    
    This is to keep the variance of the activations and gradients roughly the same across layers,
    which can help with training stability and convergence.
    
    That 2 in the numerator is specifically for ReLU activations, as it accounts for the fact 
    that ReLU outputs are zero for half of the input values on average, effectively halving the variance.
    """
    
    # conv1
    filters1 = np.random.randn(
        32,
        1,
        3,
        3
    ) * np.sqrt(2 / (1 * 3 * 3))

    bias1 = np.zeros(32)

    # conv2
    filters2 = np.random.randn(
        64,
        32,
        3,
        3
    ) * np.sqrt(2 / (32 * 3 * 3))

    bias2 = np.zeros(64)

    # dense1
    w3 = np.random.randn(
        64 * 5 * 5,
        128
    ) * np.sqrt(2 / (64 * 5 * 5))

    b3 = np.zeros((1, 128))

    # output layer
    w4 = np.random.randn(
        128,
        10
    ) * np.sqrt(2 / 128)

    b4 = np.zeros((1, 10))

# ALL WEIGHT AND DATA SAVING MECHANISM GENERATED BY AI - 2023-EE-168 (Abdullah Chaudhary) 
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "epoch_time": []
}

batch_loss_log = {
    "epoch": [],
    "batch": [],
    "loss": []
}

# =========================================================
# TRAINING
# =========================================================
def train(epochs=3, learning_rate=0.001, batch_size=32):
    global filters1, bias1, filters2, bias2, w3, b3, w4, b4
    
    num_batches = len(x_train) // batch_size

    print(num_batches, "batches per epoch")

    import time

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        total_correct = 0

        for batch in range(num_batches):
            if batch % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}/{num_batches}")

            start = batch * batch_size
            end = start + batch_size

            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            # =================================================
            # FORWARD PASS
            # =================================================

            # conv1
            z1, image_cols1 = conv_forward(
                x_batch,
                filters1,
                bias1
            )

            a1 = relu(z1)

            # pool1
            p1, max_pos1 = maxpool_forward(a1)

            # conv2
            z2, image_cols2 = conv_forward(
                p1,
                filters2,
                bias2
            )

            a2 = relu(z2)

            # pool2
            p2, max_pos2 = maxpool_forward(a2)

            # flatten
            flat = p2.reshape(batch_size, -1)

            # dense1
            z3 = flat @ w3 + b3

            a3 = relu(z3)

            # output
            z4 = a3 @ w4 + b4

            pred = softmax(z4)

            # =================================================
            # LOSS + ACCURACY
            # =================================================

            loss = cross_entropy(
                pred,
                y_batch
            )

            total_loss += loss

            predicted_classes = np.argmax(
                pred,
                axis=1
            )

            total_correct += np.sum(
                predicted_classes == y_batch
            )

            # =================================================
            # BACKWARD PASS
            # =================================================

            d_out = pred.copy()

            d_out[
                np.arange(batch_size),
                y_batch
            ] -= 1

            d_out /= batch_size

            # output layer
            d_w4 = a3.T @ d_out

            d_b4 = np.sum(
                d_out,
                axis=0,
                keepdims=True
            )

            d_a3 = d_out @ w4.T

            # relu
            d_z3 = relu_backward(
                d_a3,
                z3
            )

            # dense1
            d_w3 = flat.T @ d_z3

            d_b3 = np.sum(
                d_z3,
                axis=0,
                keepdims=True
            )

            d_flat = d_z3 @ w3.T

            # unflatten
            d_p2 = d_flat.reshape(
                p2.shape
            )

            # pool2 backward
            d_a2 = maxpool_backward(
                d_p2,
                max_pos2,
                a2.shape
            )

            # relu
            d_z2 = relu_backward(
                d_a2,
                z2
            )

            # conv2 backward
            d_p1, d_filters2, d_bias2 = conv_backward(
                d_z2,
                image_cols2,
                filters2,
                p1.shape
            )

            # pool1 backward
            d_a1 = maxpool_backward(
                d_p1,
                max_pos1,
                a1.shape
            )

            # relu
            d_z1 = relu_backward(
                d_a1,
                z1
            )

            # conv1 backward
            _, d_filters1, d_bias1 = conv_backward(
                d_z1,
                image_cols1,
                filters1,
                x_batch.shape
            )

            # =================================================
            # UPDATE WEIGHTS
            # =================================================

            filters1 -= learning_rate * d_filters1
            bias1 -= learning_rate * d_bias1

            filters2 -= learning_rate * d_filters2
            bias2 -= learning_rate * d_bias2

            w3 -= learning_rate * d_w3
            b3 -= learning_rate * d_b3

            w4 -= learning_rate * d_w4
            b4 -= learning_rate * d_b4

            if batch % 100 == 0:
                np.savez(
                    WEIGHTS_FILE,
                    filters1=filters1,
                    bias1=bias1,
                    filters2=filters2,
                    bias2=bias2,
                    w3=w3,
                    b3=b3,
                    w4=w4,
                    b4=b4
                )
                print(f"Weights successfully saved to {WEIGHTS_FILE}")

            if batch % 50 == 0:
                batch_loss_log["epoch"].append(epoch + 1)
                batch_loss_log["batch"].append(batch)
                batch_loss_log["loss"].append(loss)

        # =====================================================
        # TRAIN METRICS
        # =====================================================

        train_loss = total_loss / num_batches

        train_accuracy = total_correct / (
            num_batches * batch_size
        )


        # =====================================================
        # VALIDATION (batched)
        # =====================================================

        val_batch_size = 256
        val_preds_all = []

        for i in range(0, len(x_val), val_batch_size):
            x_vb = x_val[i:i+val_batch_size]

            z1_vb, _ = conv_forward(x_vb, filters1, bias1)
            a1_vb = relu(z1_vb)
            p1_vb, _ = maxpool_forward(a1_vb)

            z2_vb, _ = conv_forward(p1_vb, filters2, bias2)
            a2_vb = relu(z2_vb)
            p2_vb, _ = maxpool_forward(a2_vb)

            flat_vb = p2_vb.reshape(len(x_vb), -1)
            z3_vb = flat_vb @ w3 + b3
            a3_vb = relu(z3_vb)
            z4_vb = a3_vb @ w4 + b4
            pred_vb = softmax(z4_vb)

            val_preds_all.append(pred_vb)

        pred_val = np.concatenate(val_preds_all, axis=0)

        val_loss = cross_entropy(pred_val, y_val)
        val_predicted = np.argmax(pred_val, axis=1)
        val_accuracy = np.mean(val_predicted == y_val)

        epoch_time = time.time() - epoch_start

        # append to history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)
        history["epoch_time"].append(round(epoch_time, 2))

        log_msg = (
            f"Epoch {epoch+1}\n"
            f" | Train Loss: {train_loss:.4f}\n"
            f" | Train Acc: {train_accuracy:.4f}\n"
            f" | Val Loss: {val_loss:.4f}\n"
            f" | Val Acc: {val_accuracy:.4f}\n"
            f" | Time: {epoch_time:.1f}s\n\n"
        )
        print(log_msg)
        # with open("training_log.txt", "a") as f:
        #     f.write(log_msg + "\n")

# =========================================================
# TESTING (batched)
# =========================================================
# E168
def test():
    test_batch_size = 256
    test_preds_all = []

    for i in range(0, len(x_test), test_batch_size):
        x_tb = x_test[i:i+test_batch_size]

        z1_tb, _ = conv_forward(x_tb, filters1, bias1)
        a1_tb = relu(z1_tb)
        p1_tb, _ = maxpool_forward(a1_tb)

        z2_tb, _ = conv_forward(p1_tb, filters2, bias2)
        a2_tb = relu(z2_tb)
        p2_tb, _ = maxpool_forward(a2_tb)

        flat_tb = p2_tb.reshape(len(x_tb), -1)
        z3_tb = flat_tb @ w3 + b3
        a3_tb = relu(z3_tb)
        z4_tb = a3_tb @ w4 + b4
        pred_tb = softmax(z4_tb)

        test_preds_all.append(pred_tb)

    pred_test = np.concatenate(test_preds_all, axis=0)

    test_predicted = np.argmax(pred_test, axis=1)
    test_accuracy = np.mean(test_predicted == y_test)

    print("\nFinal Test Accuracy:", test_accuracy)

    # save to CSVs (Disabled for now to save time and space, can be re-enabled if needed)
    # pd.DataFrame(history).to_csv("epoch_history.csv", index=False)
    # pd.DataFrame(batch_loss_log).to_csv("batch_loss_log.csv", index=False)

    # save final test accuracy too
    # with open("training_log.txt", "a") as f:
    #     f.write(f"\nFinal Test Accuracy: {test_accuracy:.4f}\n")

    # print("All logs saved.")

    # after the epoch loop ends, before testing
    np.savez(
        WEIGHTS_FILE,
        filters1=filters1, bias1=bias1,
        filters2=filters2, bias2=bias2,
        w3=w3, b3=b3,
        w4=w4, b4=b4
    )
    print("Final weights saved.")
    


if __name__ == "__main__":
    choice = input("\n\nEnter 'train' to train the model or 'test' to evaluate on test set\nEnter 'both' to do both: ").strip().lower()
    if choice == "train":
        print("\nStarting training...\n")
        train()
    elif choice == "test":
        print("\nEvaluating on test set...\n")
        test()
    elif choice == "both":
        print("\nStarting training...\n")
        train()
        print("\nEvaluating on test set...\n")
        test()
    else:
        print("\nInvalid choice.")