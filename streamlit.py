import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import joblib

# Load model and data
# Ensure your trained model is saved as 'breast_cancer_model.pkl' using joblib in your training script
model = joblib.load('breast_cancer_model.pkl')

# Load the dataset to get feature names
data = load_breast_cancer()
feature_names = data.feature_names

# App title
st.title("Breast Cancer Prediction App")

# App subtitle
st.write("Enter the required values for each feature below to get a prediction.")

# User input
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Prepare input for prediction
input_array = np.array(list(user_input.values())).reshape(1, -1)

# Ensure input is scaled using the same scaler used during training
scaler = joblib.load('scaler.pkl')  # Make sure the scaler is saved during training
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    st.write("Prediction: **Positive**" if prediction[0] == 1 else "Prediction: **Negative**")
    st.write(f"Prediction Confidence: {np.max(prediction_proba) * 100:.2f}%")

