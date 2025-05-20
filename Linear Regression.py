# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")

# -----------------------------
# 2. Feature Selection
# -----------------------------
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
target = 'Exam_Score'

X = df[features].values
y = df[target].values.reshape(-1, 1)

# -----------------------------
# 3. Feature Normalization
# -----------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# -----------------------------
# 4. Add Bias Term
# -----------------------------
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# -----------------------------
# 5. Linear Regression with Diagnostics
# -----------------------------
def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    loss_history = []

    for epoch in range(epochs):
        y_pred = X @ weights
        error = y_pred - y
        mse = np.mean(error ** 2)
        loss_history.append(mse)

        gradient = (2 / n_samples) * X.T @ error
        weights -= learning_rate * gradient

        # Detailed diagnostics
        if epoch % 100 == 0:
            gradient_norm = np.linalg.norm(gradient)
            print(f"Epoch {epoch}: MSE = {mse:.6f}, Gradient Norm = {gradient_norm:.6f}")

    return weights, loss_history

# -----------------------------
# 6. Train the Model
# -----------------------------
weights, loss_history = linear_regression_gradient_descent(
    X_b, y, learning_rate=0.001, epochs=1000
)

# -----------------------------
# 7. Make Predictions
# -----------------------------
y_pred = X_b @ weights

print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y[i][0]}")

# -----------------------------
# 8. Analyze Loss
# -----------------------------
print(f"\nMSE First Epoch: {loss_history[0]:.6f}")
print(f"MSE Last  Epoch: {loss_history[-1]:.6f}")

# -----------------------------
# 9. Plot the Loss Curve
# -----------------------------
plt.plot(range(len(loss_history)), loss_history, marker='.')
plt.title("Linear Regression Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.show()
