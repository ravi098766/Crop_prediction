"""
Crop Recommendation Model Helper Module
Provides clean interface for loading crop recommendation model and running predictions.
"""
import pickle
import pandas as pd

MODEL_PATH = 'crop_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_crop(N, P, K, temp, humid, ph, rain):
    """
    Predicts the optimal crop given soil & climate metrics.
    Inputs:
        N: Nitrogen ratio in soil (kg/ha)
        P: Phosphorous ratio in soil (kg/ha)
        K: Potassium ratio in soil (kg/ha)
        temp: Temperature in degree Celsius
        humid: Relative humidity in %
        ph: pH value of soil (0-14)
        rain: Rainfall in mm
    Returns:
        Predicted crop label string
    """
    model = load_model()
    features_df = pd.DataFrame([[N, P, K, temp, humid, ph, rain]], 
                               columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(features_df)
    return prediction[0]

if __name__ == '__main__':
    # Simple test run
    sample_crop = predict_crop(90, 42, 43, 20.8, 82.0, 6.5, 202.9)
    print(f"Sample Prediction Test: {sample_crop} (Expected: rice)")
