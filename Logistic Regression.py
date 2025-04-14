# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------

# Read the dataset containing student performance information
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")


# -----------------------------
# 2. Create Binary Classification Target
# -----------------------------

# Define a new binary target column: Pass = 1 if Exam_Score >= 70, else Fail = 0
df['Pass'] = df['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

# -----------------------------
# 3. Feature Selection
# -----------------------------

# Select relevant features for the prediction
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
target = 'Pass'

# Extract input features (X) and binary target (y)
X = df[features].values                  # Shape: (n_samples, n_features)
y = df[target].values.reshape(-1, 1)     # Shape: (n_samples, 1)

# -----------------------------
# 4. Normalize Features
# -----------------------------

# Compute mean and standard deviation for each feature
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# Apply standardization (zero mean, unit variance)
X_normalized = (X - X_mean) / X_std

# -----------------------------
# 5. Add Bias Term
# -----------------------------

# Add a column of ones to the feature matrix to include the bias term (intercept)
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# -----------------------------
# 6. Define Sigmoid Activation Function
# -----------------------------

def sigmoid(z):
    """
    Compute the sigmoid of z.

    Parameters:
        z (ndarray): Input array

    Returns:
        ndarray: Element-wise sigmoid values
    """
    return 1 / (1 + np.exp(-z))

# -----------------------------
# 7. Logistic Regression with Gradient Descent
# -----------------------------

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    """
    Train logistic regression using batch gradient descent.

    Parameters:
        X (ndarray): Input features with bias term, shape (n_samples, n_features+1)
        y (ndarray): Binary target values, shape (n_samples, 1)
        learning_rate (float): Learning rate for weight updates
        epochs (int): Number of iterations

    Returns:
        weights (ndarray): Learned weights
        loss_history (list): Binary cross-entropy loss over epochs
    """
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))  # Initialize weights to zero
    loss_history = []                    # Store loss for each epoch

    for epoch in range(epochs):
        linear_output = X @ weights      # Linear combination of inputs and weights
        y_pred = sigmoid(linear_output)  # Apply sigmoid to get predicted probabilities

        error = y_pred - y               # Compute prediction error

        # Binary cross-entropy loss calculation
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        loss_history.append(loss)

        # Compute gradient
        gradient = X.T @ error / n_samples

        # Update weights
        weights -= learning_rate * gradient

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return weights, loss_history

# -----------------------------
# 8. Train the Logistic Model
# -----------------------------

# Train logistic regression model using gradient descent
weights, loss_history = logistic_regression(X_b, y, learning_rate=0.1, epochs=1000)

# -----------------------------
# 9. Make Predictions and Evaluate
# -----------------------------

# Compute predicted probabilities using the sigmoid function
y_prob = sigmoid(X_b @ weights)

# Convert probabilities to binary predictions using a threshold of 0.5
y_pred = (y_prob >= 0.5).astype(int)

# Compute classification accuracy
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Display first 5 predictions compared to actual labels
print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]}, Actual: {y[i][0]}")

# -----------------------------
# 10. Plot Loss Curve
# -----------------------------

# Visualize the binary cross-entropy loss over training epochs
plt.plot(loss_history)
plt.title("Logistic Regression Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
