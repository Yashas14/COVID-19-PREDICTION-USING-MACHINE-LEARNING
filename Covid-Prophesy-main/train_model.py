"""
Train the COVID-19 prediction model and save it as model.pkl.
Run this script whenever you need to retrain or regenerate the model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
covid = pd.read_csv('Covid Dataset.csv')
print(f"Dataset loaded: {covid.shape[0]} rows, {covid.shape[1]} columns")

# Encode all categorical columns
e = LabelEncoder()
for col in covid.columns:
    covid[col] = e.fit_transform(covid[col])

# Drop low-correlation columns (as identified in analysis)
cols_to_drop = [
    'Running Nose', 'Chronic Lung Disease', 'Headache',
    'Heart Disease', 'Diabetes', 'Gastrointestinal ',
    'Wearing Masks', 'Sanitization from Market', 'Asthma', 'Fatigue '
]
covid = covid.drop(columns=[c for c in cols_to_drop if c in covid.columns])

print(f"Features used: {list(covid.drop('target', axis=1).columns)}")

# Split features and target
X = covid.drop('target', axis=1)
y = covid['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f"Model accuracy: {accuracy:.2f}%")

# Save as model.pkl (used by the Flask app)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl saved successfully.")
