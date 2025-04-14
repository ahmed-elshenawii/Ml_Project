# -----------------------------------------------
# LINEAR REGRESSION USING SCIKIT-LEARN
# -----------------------------------------------
# This script demonstrates three different ways to perform Linear Regression:
# 1. Basic Linear Regression (Least Squares)
# 2. Ridge Regression (L2 Regularization)
# 3. Linear Regression via Gradient Descent (SGDRegressor)

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -----------------------------
# 2. Load Dataset
# -----------------------------

# Load the student performance dataset
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")


# Preview the first few rows of the dataset
print("Sample of the dataset:")
print(df.head())

# -----------------------------
# 3. Feature and Target Selection
# -----------------------------

# Select independent variables (features)
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
X = df[features].values

# Select target variable (what we want to predict)
y_linear = df['Exam_Score'].values.reshape(-1, 1)  # Reshape to column vector

# -----------------------------
# 4. Feature Normalization
# -----------------------------

# Standardize features (mean = 0, std = 1) for gradient-based methods
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Apply Different Linear Regression Methods
# -----------------------------

print("\n=== LINEAR REGRESSION MODELS ===")

# --- 1. Basic Linear Regression (No Regularization) ---
model = LinearRegression()
model.fit(X, y_linear)
predictions = model.predict(X)
mse_basic = mean_squared_error(y_linear, predictions)
print("MSE (Basic Linear Regression):", mse_basic)

# --- 2. Ridge Regression (L2 Regularization) ---
ridge_model = Ridge(alpha=1.0)  # alpha = regularization strength
ridge_model.fit(X, y_linear)
ridge_preds = ridge_model.predict(X)
mse_ridge = mean_squared_error(y_linear, ridge_preds)
print("MSE (Ridge Regularization):", mse_ridge)

# --- 3. Linear Regression via Gradient Descent ---
# Note: We use normalized features for gradient-based methods

# Variant 1: Batch-like Gradient Descent (with 'invscaling' learning rate)
sgd_batch = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01)
sgd_batch.fit(X_scaled, y_linear.ravel())  # Flatten target array for sklearn
mse_batch = mean_squared_error(y_linear, sgd_batch.predict(X_scaled))
print("MSE (Gradient Descent - Batch):", mse_batch)

# Variant 2: Stochastic Gradient Descent (with constant learning rate)
sgd_sgd = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
sgd_sgd.fit(X_scaled, y_linear.ravel())
mse_sgd = mean_squared_error(y_linear, sgd_sgd.predict(X_scaled))
print("MSE (Gradient Descent - SGD):", mse_sgd)

# -----------------------------
# 6. Summary of Results
# -----------------------------
print("\n=== SUMMARY ===")
print(f"Basic Linear Regression MSE:      {mse_basic:.2f}")
print(f"Ridge Linear Regression MSE:      {mse_ridge:.2f}")
print(f"Gradient Descent (Batch) MSE:     {mse_batch:.2f}")
print(f"Gradient Descent (SGD) MSE:       {mse_sgd:.2f}")
