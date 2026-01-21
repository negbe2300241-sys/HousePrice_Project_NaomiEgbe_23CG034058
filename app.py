from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- LOADING MODEL & SCALER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ⚠️ Using 'Model' (Capital M) as requested
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'house_price_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Model', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Loaded files from {BASE_DIR}/Model/")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    model = None
    scaler = None

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return "Error: Model or Scaler not loaded.", 500

    try:
        # Get values from HTML form
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        area = float(request.form['area'])

        # Create array and scale
        # ⚠️ Double brackets [[ ]] are needed for sklearn
        features = np.array([[bedrooms, bathrooms, area]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        
        # Format result as currency (e.g., $150,000.00)
        result_text = f"${prediction[0]:,.2f}"

        return render_template('index.html', prediction_text=f'Estimated House Price: {result_text}')

    except Exception as e:
        return f"Prediction Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)