# Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset

df = pd.read_csv("ChurnDS.csv")

# Display first 5 rows

print(df.head())

# Drop Customer ID

df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Encode categorical columns

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'str':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

print(df.dtypes)


# Features and Target

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model

lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)

# Predictions

y_pred = lr.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))