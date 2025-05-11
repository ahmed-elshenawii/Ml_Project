import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")

# Convert score to categories
def categorize(score):
    if score < 60:
        return "Low"
    elif score <= 75:
        return "Medium"
    else:
        return "High"

df["Score_Category"] = df["Exam_Score"].apply(categorize)
df.drop(columns=["Exam_Score"], inplace=True)

# Encode features and labels
X = pd.get_dummies(df.drop("Score_Category", axis=1))
y = LabelEncoder().fit_transform(df["Score_Category"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train polynomial kernel SVM
svm_poly = SVC(kernel='poly', degree=2, C=1.0)
svm_poly.fit(X_train, y_train)
y_pred = svm_poly.predict(X_test)

# Accuracy
acc_poly = accuracy_score(y_test, y_pred)
print("Polynomial Kernel SVM Accuracy:", round(acc_poly * 100, 2), "%")

# Visualization
X_2d = PCA(n_components=2).fit_transform(X_scaled)
svm_poly_2d = SVC(kernel='poly', degree=2, C=1.0)
svm_poly_2d.fit(X_2d, y)

# Mesh grid for boundary
h = .02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_poly_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.title("Polynomial Kernel SVM (Degree 2)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()
