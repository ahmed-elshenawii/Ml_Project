# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------

# Read the CSV file containing student performance data
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")

# -----------------------------
# 2. Feature Selection
# -----------------------------

# Define the features (independent variables)
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']

# Define the target (dependent variable)
target = 'Exam_Score'

# Extract features and target from the DataFrame
X = df[features].values                  # Shape: (n_samples, n_features)
y = df[target].values.reshape(-1, 1)     # Shape: (n_samples, 1)

# -----------------------------
# 3. Feature Normalization
# -----------------------------

# Compute the mean and standard deviation of each feature
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# Standardize features (mean = 0, std = 1)
X_normalized = (X - X_mean) / X_std

# -----------------------------
# 4. Add Bias Term
# -----------------------------

# Add a column of 1s to X to account for the bias (intercept) in the linear model
# Shape becomes (n_samples, n_features + 1)
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# -----------------------------
# 5. Linear Regression Using Gradient Descent
# -----------------------------

def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Perform linear regression using batch gradient descent.

    Parameters:
        X (ndarray): Input features with bias term, shape (n_samples, n_features+1)
        y (ndarray): Target values, shape (n_samples, 1)
        learning_rate (float): Step size for gradient update
        epochs (int): Number of iterations

    Returns:
        weights (ndarray): Learned weights, shape (n_features+1, 1)
        loss_history (list): MSE loss over epochs
    """
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))  # Initialize weights to zero
    loss_history = []                    # Store MSE loss for each epoch

    for epoch in range(epochs):
        y_pred = X @ weights             # Compute predictions
        error = y_pred - y               # Compute prediction error
        mse = np.mean(error ** 2)        # Mean Squared Error (MSE)
        loss_history.append(mse)

        # Compute gradient of the loss w.r.t. weights
        gradient = (2 / n_samples) * X.T @ error

        # Update weights in the direction of negative gradient
        weights -= learning_rate * gradient

        # Print MSE every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE = {mse:.4f}")

    return weights, loss_history

# -----------------------------
# 6. Train the Model
# -----------------------------

# Train the model using gradient descent
weights, loss_history = linear_regression_gradient_descent(
    X_b, y, learning_rate=0.1, epochs=1000
)

# -----------------------------
# 7. Make Predictions
# -----------------------------

# Use the trained model to make predictions on the training data
y_pred = X_b @ weights

# Print the first 5 predictions and actual exam scores
print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y[i][0]}")

# -----------------------------
# 8. Plot the Loss Curve
# -----------------------------

# Visualize how the MSE changed over epochs
plt.plot(loss_history)
plt.title("Linear Regression Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.show()
