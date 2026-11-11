
# Step 1: Import Pandas
import pandas as pd

# Step 2: Define column names
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Step 3: Load the dataset
df = pd.read_csv(
    "dataset/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

# Step 4: Display first 5 rows
print("First 5 rows:")
print(df.head())

# Step 5: Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Step 6: Display column information
print("\nDataset Information:")
print(df.info())

# Step 7: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 8: Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Step 9: Target Variable Distribution

print("\nHeart Disease Distribution:")

print(df["target"].value_counts().sort_index())

# Step 10: Convert target into binary classification

df["target"] = (df["target"] > 0).astype(int)

print("\nBinary Target Distribution:")
print(df["target"].value_counts())

# Step 11: Visualize Target Distribution

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="target")

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.savefig("target_distribution.png")
plt.show()

# Step 12: Correlation Heatmap

plt.figure(figsize=(12, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# Step 13: Handle Missing Values

# Replace missing values in ca with mode
df["ca"] = df["ca"].fillna(df["ca"].mode()[0])

# Replace missing values in thal with mode
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# Check missing values again
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Step 14: Separate Features and Target

# X contains independent variables
X = df.drop("target", axis=1)

# y contains dependent variable
y = df["target"]

# Display shapes
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Step 15: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset shapes
print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

# Step 16: Feature Scaling

from sklearn.preprocessing import StandardScaler

# Create StandardScaler object
scaler = StandardScaler()

# Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

# Display scaled data
print("\nScaled Training Data:")
print(X_train_scaled[:5])

print("\nScaled Testing Data:")
print(X_test_scaled[:5])


# Step 17: Logistic Regression Model

from sklearn.linear_model import LogisticRegression

# Create model
log_model = LogisticRegression(random_state=42)

# Train model
log_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_model.predict(X_test_scaled)

# Display predictions
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Check model accuracy
print("\nModel Accuracy:")
print(log_model.score(X_test_scaled, y_test))

# Step 18: Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot(cmap="Blues")

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Step 19: Classification Evaluation Metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

# Display results
print("\nModel Evaluation Results")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test_scaled)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Model Accuracy:", rf_accuracy)
print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred))

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))
import joblib

# Save the Random Forest model
joblib.dump(rf_model, "heart_disease_model.pkl")

print("\nModel saved successfully!")
# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")

print("Scaler saved successfully!")
# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")

print("Scaler saved successfully!")