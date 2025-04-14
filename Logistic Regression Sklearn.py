# =======================================================
# LOGISTIC REGRESSION USING SCIKIT-LEARN (4 Approaches)
# =======================================================
# This script demonstrates how to use different logistic regression methods:
# 1. Basic Logistic Regression (no regularization)
# 2. Logistic Regression with L2 Regularization (Ridge)
# 3. Gradient Descent using SGDClassifier (Batch)
# 4. Stochastic Gradient Descent (SGD) using SGDClassifier

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")


# -----------------------------
# 3. Create Binary Target Variable
# -----------------------------
# Define "Pass" as Exam_Score >= 60
df['Pass'] = df['Exam_Score'].apply(lambda x: 1 if x >= 60 else 0)

# -----------------------------
# 4. Select Features and Target
# -----------------------------
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
X = df[features].values
y = df['Pass'].values

# -----------------------------
# 5. Normalize Features
# -----------------------------
# Important for gradient descent convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 6. Apply Different Logistic Regression Models
# -----------------------------
print("=== LOGISTIC REGRESSION MODELS ===")

# --- 1. Basic Logistic Regression (No Regularization) ---
# 'penalty="none"' disables regularization
basic_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
basic_model.fit(X, y)
basic_preds = basic_model.predict(X)
acc_basic = accuracy_score(y, basic_preds)
print("Accuracy (Basic Logistic Regression):", acc_basic)

# --- 2. Logistic Regression with L2 Regularization (Ridge) ---
# Adds a penalty to reduce overfitting
ridge_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
ridge_model.fit(X, y)
ridge_preds = ridge_model.predict(X)
acc_ridge = accuracy_score(y, ridge_preds)
print("Accuracy (Ridge Regularization):", acc_ridge)

# --- 3. Gradient Descent (Batch Mode) ---
# 'invscaling' gradually decreases the learning rate
sgd_batch = SGDClassifier(loss='log_loss', learning_rate='invscaling', eta0=0.01, max_iter=1000)
sgd_batch.fit(X_scaled, y)
batch_preds = sgd_batch.predict(X_scaled)
acc_batch = accuracy_score(y, batch_preds)
print("Accuracy (Gradient Descent - Batch):", acc_batch)

# --- 4. Stochastic Gradient Descent (SGD) ---
# Uses constant learning rate (fixed)
sgd_sgd = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, max_iter=1000)
sgd_sgd.fit(X_scaled, y)
sgd_preds = sgd_sgd.predict(X_scaled)
acc_sgd = accuracy_score(y, sgd_preds)
print("Accuracy (Gradient Descent - SGD):", acc_sgd)

# -----------------------------
# 7. Summary of Model Results
# -----------------------------
print("\n=== SUMMARY ===")
print(f"Basic Logistic Regression Accuracy:      {acc_basic:.2f}")
print(f"Ridge Logistic Regression Accuracy:      {acc_ridge:.2f}")
print(f"Gradient Descent (Batch) Accuracy:       {acc_batch:.2f}")
print(f"Gradient Descent (SGD) Accuracy:         {acc_sgd:.2f}")
