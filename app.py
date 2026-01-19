import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("House Price Prediction")

# Input fields
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)

# Predict button
if st.button("Predict Price"):
    # Create input array
    X = np.array([[bedrooms, bathrooms, area]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
