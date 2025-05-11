import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")

# Categorize Exam_Score into classes
def categorize_score(score):
    if score < 60:
        return 'Low'
    elif score <= 75:
        return 'Medium'
    else:
        return 'High'

df['Score_Category'] = df['Exam_Score'].apply(categorize_score)
df.drop(columns=['Exam_Score'], inplace=True)

# Separate features and target
X = df.drop(columns=['Score_Category'])
y = df['Score_Category']

# Encode categorical features
X_encoded = pd.get_dummies(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train SVM with quadratic kernel
svm_poly = SVC(kernel='poly', degree=2, C=1.0)
svm_poly.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_poly.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with Quadratic Kernel SVM:", round(accuracy * 100, 2), "%")

# Visualization using PCA (2D projection)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Train SVM with quadratic kernel on 2D data
svm_quadratic = SVC(kernel='poly', degree=2, C=1.0)
svm_quadratic.fit(X_2d, y_encoded)

# Create a mesh for plotting decision boundary
h = .02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for mesh grid
Z_quadratic = svm_quadratic.predict(np.c_[xx.ravel(), yy.ravel()])
Z_quadratic = Z_quadratic.reshape(xx.shape)

# Plotting
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z_quadratic, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('Quadratic Kernel SVM (Kernel Trick)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.show()
