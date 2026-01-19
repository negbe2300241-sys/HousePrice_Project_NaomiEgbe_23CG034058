import streamlit as st
import numpy as np
import joblib
import os

st.title("🏠 House Price Prediction System")

# Function to load model safely
@st.cache_resource
def load_resources():
    # Check if files exist before loading
    if not os.path.exists("model/house_price_model.pkl") or not os.path.exists("model/scaler.pkl"):
        st.error("Model files not found! Please check the 'model' directory.")
        return None, None
    
    try:
        loaded_model = joblib.load("model/house_price_model.pkl")
        loaded_scaler = joblib.load("model/scaler.pkl")
        return loaded_model, loaded_scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_resources()

# Only show the inputs if the model loaded successfully
if model and scaler:
    st.write("Enter house details to predict the price:")

    overall_qual = st.number_input("Overall Quality (1–10)", 1, 10, 5)
    gr_liv_area = st.number_input("Ground Living Area (sq ft)", 300, 6000, 1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 5000, 800)
    garage_cars = st.number_input("Garage Capacity (cars)", 0, 4, 2)
    full_bath = st.number_input("Number of Full Bathrooms", 0, 4, 2)
    year_built = st.number_input("Year Built", 1870, 2025, 2005)

    if st.button("Predict Price"):
        input_data = np.array([[ 
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            full_bath,
            year_built
        ]])

        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            st.success(f"🏷️ Estimated House Price: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Prediction Error: {e}. Check if your feature count matches the model.")