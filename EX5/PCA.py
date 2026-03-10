# ==============================
# PCA Implementation on
# Breast Cancer Wisconsin Dataset
# ==============================
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# ------------------------------
# 1. Load Dataset
# ------------------------------
print("Loading dataset...")
df = pd.read_csv("data.csv")
print("Original Dataset Shape:", df.shape)
print(df.head())
# ------------------------------
# 2. Data Preprocessing
# ------------------------------
# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)# Convert target variable to numeric
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
# Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nData Standardized.")
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
# Plot Explained Variance
plt.figure()
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid()
plt.show()
# Print variance values
print("\nExplained Variance Ratio:")
print(pca_full.explained_variance_ratio_)
pca = PCA(n_components=0.95) # Automatically selects components
X_pca = pca.fit_transform(X_scaled)
print("\nOriginal Shape:", X_scaled.shape)
print("Reduced Shape after PCA:", X_pca.shape)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Visualization")
plt.show()
# ------------------------------
# 7. Model WITHOUT PCA
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n==============================")
print("Results WITHOUT PCA")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
X_pca, y, test_size=0.2, random_state=42)
model_pca = RandomForestClassifier(random_state=42)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
print("\n==============================")
print("Results WITH PCA")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_pca))
print(classification_report(y_test, y_pred_pca))
print("\nProgram Completed Successfully ")
