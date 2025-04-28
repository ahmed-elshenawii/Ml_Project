import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset and Create Target
# -----------------------------
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")
df['Pass'] = df['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)
df['Pass'] = df['Pass'].replace(0, -1)

# -----------------------------
# 2. Feature Selection
# -----------------------------
features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores']
target = 'Pass'
X = df[features].values
y = df[target].values

# -----------------------------
# 3. Normalize Features Manually
# -----------------------------
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_normalized = (X - mean) / std

# -----------------------------
# 4. Split Dataset into Training and Test Sets
# -----------------------------
m = X_normalized.shape[0]
indices = np.arange(m)
np.random.shuffle(indices)
split_index = int(0.8 * m)
train_idx = indices[:split_index]
test_idx = indices[split_index:]

X_train = X_normalized[train_idx]
y_train = y[train_idx]
X_test = X_normalized[test_idx]
y_test = y[test_idx]

# -----------------------------
# 5. Define Kernel Functions
# -----------------------------
def linear_kernel(x1, x2, gamma=None):
    """Compute the linear kernel between two vectors."""
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma):
    """Compute the RBF (Gaussian) kernel between two vectors."""
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# -----------------------------
# 6. Implement the SVM Classifier with Pre-computed Kernel
# -----------------------------
class SVMClassifier:
    def __init__(self, kernel, C=1.0, tol=1e-3, max_passes=5, gamma=0.1):
        self.kernel = kernel      # Kernel function (e.g., linear_kernel or rbf_kernel)
        self.C = C                # Regularization parameter
        self.tol = tol            # Tolerance for KKT conditions
        self.max_passes = max_passes  # Max iterations without alpha changes
        self.gamma = gamma

    def fit(self, X, y):
        self.X = X
        self.y = y
        m = X.shape[0]
        self.alphas = np.zeros(m)
        self.b = 0.0

        # Pre-compute the full kernel matrix
        self.K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                self.K[i, j] = self.kernel(X[i], X[j], self.gamma)

        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                # Use pre-computed kernel values for sample i
                f_i = np.sum(self.alphas * self.y * self.K[i, :]) + self.b
                E_i = f_i - y[i]
                
                if ((y[i] * E_i < -self.tol and self.alphas[i] < self.C) or 
                    (y[i] * E_i > self.tol and self.alphas[i] > 0)):
                    
                    # Randomly pick j != i
                    j = i
                    while j == i:
                        j = np.random.randint(0, m)
                    
                    f_j = np.sum(self.alphas * self.y * self.K[j, :]) + self.b
                    E_j = f_j - y[j]
                    
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute L and H
                    if y[i] == y[j]:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    else:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    self.alphas[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha[i]
                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update bias term b
                    b1 = self.b - E_i - y[i]*(self.alphas[i]-alpha_i_old)*self.K[i, i] - y[j]*(self.alphas[j]-alpha_j_old)*self.K[i, j]
                    b2 = self.b - E_j - y[i]*(self.alphas[i]-alpha_i_old)*self.K[i, j] - y[j]*(self.alphas[j]-alpha_j_old)*self.K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        return self

    def project(self, X):
        """Compute the decision function for samples in X."""
        m = X.shape[0]
        result = np.zeros(m)
        for i in range(m):
            # Compute kernel between training set and the sample X[i]
            K_eval = np.array([self.kernel(self.X[j], X[i], self.gamma) for j in range(self.X.shape[0])])
            result[i] = np.sum(self.alphas * self.y * K_eval) + self.b
        return result

    def predict(self, X):
        """Predict class labels (+1 or -1) for samples in X."""
        return np.sign(self.project(X))

# -----------------------------
# 7. Train the SVM Model
# -----------------------------
# Here, we use the linear kernel for demonstration. (Switch to rbf_kernel if needed)
svm_model = SVMClassifier(kernel=linear_kernel, C=1.0, tol=1e-3, max_passes=5, gamma=0.1)
svm_model.fit(X_train, y_train)

# -----------------------------
# 8. Make Predictions and Evaluate
# -----------------------------
y_pred = svm_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nSample Predictions:")
for i in range(min(5, len(y_pred))):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")

# -----------------------------
# 9. Visualization
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(y_pred, 'bo-', label="Predicted Labels")
plt.plot(y_test, 'rx-', label="Actual Labels")
plt.title("Predicted vs Actual Labels (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Label")
plt.legend()
plt.show()