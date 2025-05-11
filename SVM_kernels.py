import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- تحميل البيانات ----------------
df = pd.read_csv("C:/Users/ahmed/OneDrive/Desktop/New folder/data/StudentPerformanceFactors.csv")

# تحويل Exam_Score إلى فئتين: ناجح جداً (>=70) وغير ناجح (<70)
df['Passed'] = (df['Exam_Score'] >= 70).astype(int)
df.drop(columns=['Exam_Score'], inplace=True)

# ملء القيم الفارغة
df.fillna(df.mode().iloc[0], inplace=True)

# فصل السمات عن الهدف
X = df.drop(columns=['Passed'])
y = df['Passed']

# الأعمدة الرقمية والفئوية
numerical_cols = X.select_dtypes(include=['int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ---------------- المعالجة المسبقة ----------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# تجهيز PCA للرسم
X_full_processed = preprocessor.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(
    X_full_processed.toarray() if hasattr(X_full_processed, "toarray") else X_full_processed
)

def plot_pca(X_pca, y, title):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
    plt.title(f"PCA Projection - {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------- Quadratic Kernel (Polynomial d=2) ----------------
quad_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svm', SVC(kernel='poly', degree=2))
])
quad_pipeline.fit(X_train, y_train)
y_pred_quad = quad_pipeline.predict(X_test)
print("\n[Quadratic Kernel SVM]")
print("Accuracy:", accuracy_score(y_test, y_pred_quad))
print(classification_report(y_test, y_pred_quad))
plot_pca(X_pca, y, "Quadratic Kernel")

# ---------------- Polynomial Kernel (d=2) ----------------
poly_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svm', SVC(kernel='poly', degree=2))
])
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)
print("\n[Polynomial Kernel SVM (Degree=2)]")
print("Accuracy:", accuracy_score(y_test, y_pred_poly))
print(classification_report(y_test, y_pred_poly))
plot_pca(X_pca, y, "Polynomial Kernel")

# ---------------- RBF Kernel ----------------
rbf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svm', SVC(kernel='rbf'))
])
rbf_pipeline.fit(X_train, y_train)
y_pred_rbf = rbf_pipeline.predict(X_test)
print("\n[RBF Kernel SVM]")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))
plot_pca(X_pca, y, "RBF Kernel")