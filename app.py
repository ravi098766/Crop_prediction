from flask import Flask, render_template, request, jsonify
import pickle
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Trained ML Model & Metadata
try:
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    metadata = {}

CROP_INFO = {
    "rice": "Requires heavy rainfall and high humidity. Ideal for clayey or loamy soil with standing water capability.",
    "maize": "Thrives in warm weather with moderate rainfall. Well-drained fertile soils are recommended.",
    "chickpea": "Requires cool weather and moderate rainfall. Grows best in well-drained silt loam soils.",
    "kidneybeans": "Prefers moderate temperature and well-drained soil rich in organic matter.",
    "pigeonpeas": "Drought-tolerant crop suitable for semi-arid tropics. Thrives in loamy to heavy soils.",
    "mothbeans": "Highly drought-resistant pulse crop suitable for dry and arid regions with sandy soil.",
    "mungbean": "Short-duration crop requiring warm temperatures and well-drained loamy soil.",
    "blackgram": "Grows well in hot and humid climates. Prefers heavy loamy soils with high water retention.",
    "lentil": "Cool-season crop requiring moderate moisture and loose loamy soil with neutral pH.",
    "pomegranate": "Thrives in semi-arid regions with warm summers and cool winters. Prefers deep loamy soil.",
    "banana": "Tropical crop requiring rich, well-drained soil, high humidity, and warm temperature.",
    "mango": "Thrives in tropical to subtropical climates with distinct dry seasons. Prefers deep, well-drained soil.",
    "grapes": "Requires dry weather during flowering and fruit ripening. Thrives in well-drained sandy loam soil.",
    "watermelon": "Warm-season crop requiring high temperature, full sun, and sandy soil with good drainage.",
    "muskmelon": "Requires warm, dry weather and well-drained sandy loamy soil rich in organic matter.",
    "apple": "Temperate fruit crop requiring cold winter temperatures and well-drained loamy soil.",
    "orange": "Subtropical crop needing well-drained loamy soil and moderate rainfall with good sunlight.",
    "papaya": "Fast-growing tropical crop requiring continuous warm weather and loose, fertile soil.",
    "coconut": "Coastal tropical crop requiring high humidity, abundant rainfall, and sandy loam soil.",
    "cotton": "Requires warm climate, sunny weather, moderate rainfall, and fertile black soil.",
    "jute": "Requires hot and humid climate with heavy rainfall. Ideal for alluvial soils in river basins.",
    "coffee": "Requires cool to warm tropical climate, humid air, shade, and rich acidic soil."
}

@app.route('/')
def home():
    return render_template('index.html', metadata=metadata)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        N = float(request.form.get('N', 0))
        P = float(request.form.get('P', 0))
        K = float(request.form.get('K', 0))
        temp = float(request.form.get('temp', 0))
        humid = float(request.form.get('humid', 0))
        ph = float(request.form.get('ph', 0))
        rain = float(request.form.get('rain', 0))

        # Input Validation Rules
        errors = []
        if N < 0 or N > 200: errors.append("Nitrogen (N) should be between 0 and 200 kg/ha.")
        if P < 0 or P > 200: errors.append("Phosphorus (P) should be between 0 and 200 kg/ha.")
        if K < 0 or K > 250: errors.append("Potassium (K) should be between 0 and 250 kg/ha.")
        if temp < 0 or temp > 60: errors.append("Temperature should be between 0°C and 60°C.")
        if humid < 0 or humid > 100: errors.append("Humidity should be between 0% and 100%.")
        if ph < 0 or ph > 14: errors.append("pH Value must be between 0 and 14.")
        if rain < 0 or rain > 1000: errors.append("Rainfall should be between 0 and 1000 mm.")

        if errors:
            return render_template('result.html', error=" | ".join(errors))

        features_df = pd.DataFrame([[N, P, K, temp, humid, ph, rain]], 
                                   columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction = model.predict(features_df)[0]
        
        # Confidence score calculation
        confidence = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_df)[0]
            confidence = round(float(np.max(probs)) * 100, 2)

        info = CROP_INFO.get(prediction.lower(), "Recommended based on soil & climate attributes.")

        return render_template('result.html', 
                               crop=prediction.capitalize(), 
                               confidence=confidence, 
                               info=info, 
                               inputs={'N': N, 'P': P, 'K': K, 'Temp': temp, 'Humid': humid, 'pH': ph, 'Rain': rain})
    except Exception as e:
        return render_template('result.html', error=f"Invalid Input: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for IoT / Mobile integration scenarios"""
    try:
        data = request.get_json(force=True)
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temp = float(data.get('temp', 0))
        humid = float(data.get('humid', 0))
        ph = float(data.get('ph', 0))
        rain = float(data.get('rain', 0))

        features_df = pd.DataFrame([[N, P, K, temp, humid, ph, rain]], 
                                   columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction = model.predict(features_df)[0]
        
        confidence = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_df)[0]
            confidence = round(float(np.max(probs)) * 100, 2)

        return jsonify({
            'status': 'success',
            'recommended_crop': prediction,
            'confidence_percentage': confidence,
            'advisory': CROP_INFO.get(prediction.lower(), "")
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
