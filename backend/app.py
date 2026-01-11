"""
Flask Backend API for Heart Failure Risk Prediction
Integrates trained XGBoost model with SHAP explanations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================
# GLOBAL VARIABLES & MODEL LOADING
# ============================================

# Feature names (must match training order)
FEATURE_NAMES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'time'
]

# Feature descriptions for human-readable explanations
FEATURE_DESCRIPTIONS = {
    "age": "older age",
    "ejection_fraction": "reduced heart pumping ability",
    "serum_creatinine": "impaired kidney function",
    "serum_sodium": "low blood sodium level",
    "anaemia": "presence of anaemia",
    "high_blood_pressure": "history of high blood pressure",
    "platelets": "platelet count",
    "diabetes": "diabetes status",
    "creatinine_phosphokinase": "elevated muscle enzyme levels",
    "sex": "biological sex",
    "time": "follow-up period"
}

# Load trained model and scaler
# NOTE: Save these during training with:
# pickle.dump(calibrated_model, open('model.pkl', 'wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # For SHAP, we need the underlying XGBoost model (before calibration)
    # If you saved best_model from your code:
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    MODEL_LOADED = True
    print("✅ Models loaded successfully!")
    
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ Model loading failed: {e}")
    print("Running in demo mode with mock predictions")

# ============================================
# HELPER FUNCTIONS
# ============================================

def interpret_shap_value(shap_val):
    """Convert SHAP value to human-readable impact description"""
    abs_val = abs(shap_val)
    if abs_val >= 0.6:
        return "strongly increases risk" if shap_val > 0 else "strongly reduces risk"
    elif abs_val >= 0.3:
        return "moderately increases risk" if shap_val > 0 else "moderately reduces risk"
    elif abs_val >= 0.1:
        return "slightly increases risk" if shap_val > 0 else "slightly reduces risk"
    else:
        return "has minimal impact"

def generate_shap_explanations(patient_data, shap_values):
    """Generate human-readable SHAP explanations"""
    explanations = []
    
    for i, feature in enumerate(FEATURE_NAMES):
        shap_val = float(shap_values[i])
        feature_value = patient_data[feature]
        
        explanations.append({
            "feature": feature,
            "value": float(feature_value),
            "impact": interpret_shap_value(shap_val),
            "shap_value": shap_val,
            "description": f"{FEATURE_DESCRIPTIONS.get(feature, feature)} {interpret_shap_value(shap_val)}"
        })
    
    # Sort by absolute SHAP value (most important first)
    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanations

def get_risk_level(probability):
    """Categorize risk level based on probability"""
    if probability < 0.4:
        return "LOW"
    elif probability < 0.7:
        return "MODERATE"
    else:
        return "HIGH"

def validate_input(data):
    """Validate input data"""
    errors = []
    
    # Check all required features are present
    for feature in FEATURE_NAMES:
        if feature not in data:
            errors.append(f"Missing required field: {feature}")
    
    # Validate numeric ranges
    validations = {
        'age': (40, 95),
        'ejection_fraction': (14, 80),
        'serum_creatinine': (0.5, 9.4),
        'serum_sodium': (113, 148),
        'time': (4, 285),
        'creatinine_phosphokinase': (23, 7861),
        'platelets': (25100, 850000),
        'anaemia': (0, 1),
        'high_blood_pressure': (0, 1),
        'diabetes': (0, 1),
        'sex': (0, 1)
    }
    
    for feature, (min_val, max_val) in validations.items():
        if feature in data:
            try:
                value = float(data[feature])
                if not (min_val <= value <= max_val):
                    errors.append(f"{feature} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{feature} must be a valid number")
    
    return errors

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expected JSON format:
    {
        "age": 75,
        "anaemia": 0,
        "creatinine_phosphokinase": 582,
        "diabetes": 0,
        "ejection_fraction": 20,
        "high_blood_pressure": 1,
        "platelets": 265000,
        "serum_creatinine": 1.9,
        "serum_sodium": 130,
        "sex": 1,
        "time": 4
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate input
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({"error": "Validation failed", "details": validation_errors}), 400
        
        # Create DataFrame with correct feature order
        patient_df = pd.DataFrame([{feature: float(data[feature]) for feature in FEATURE_NAMES}])
        
        if MODEL_LOADED:
            # ===== REAL MODEL PREDICTION =====
            
            # Scale features
            patient_scaled = scaler.transform(patient_df)
            
            # Get probability prediction from calibrated model
            death_probability = float(model.predict_proba(patient_df)[0][1])
            death_prediction = int(death_probability >= 0.5)
            
            # Generate SHAP explanations
            shap_values = explainer.shap_values(patient_scaled)[0]
            explanations = generate_shap_explanations(patient_df.iloc[0], shap_values)
            
        else:
            # ===== MOCK PREDICTION (when model not loaded) =====
            death_probability, explanations = generate_mock_prediction(patient_df.iloc[0])
            death_prediction = int(death_probability >= 0.5)
        
        # Prepare response
        risk_level = get_risk_level(death_probability)
        
        response = {
            "success": True,
            "prediction": death_prediction,
            "probability": float(death_probability),
            "risk_level": risk_level,
            "topFactors": explanations[:5],  # Top 5 factors
            "allFactors": explanations,  # All factors
            "patient_data": data,
            "model_version": "XGBoost + Calibration" if MODEL_LOADED else "Mock Model"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple patients
    Expected JSON format:
    {
        "patients": [
            {...patient1_data...},
            {...patient2_data...},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({"error": "No patients data provided"}), 400
        
        patients = data['patients']
        results = []
        
        for idx, patient_data in enumerate(patients):
            # Validate input
            validation_errors = validate_input(patient_data)
            if validation_errors:
                results.append({
                    "patient_index": idx,
                    "success": False,
                    "error": "Validation failed",
                    "details": validation_errors
                })
                continue
            
            # Create DataFrame
            patient_df = pd.DataFrame([{feature: float(patient_data[feature]) for feature in FEATURE_NAMES}])
            
            if MODEL_LOADED:
                # Real prediction
                patient_scaled = scaler.transform(patient_df)
                death_probability = float(model.predict_proba(patient_df)[0][1])
                death_prediction = int(death_probability >= 0.5)
                shap_values = explainer.shap_values(patient_scaled)[0]
                explanations = generate_shap_explanations(patient_df.iloc[0], shap_values)
            else:
                # Mock prediction
                death_probability, explanations = generate_mock_prediction(patient_df.iloc[0])
                death_prediction = int(death_probability >= 0.5)
            
            results.append({
                "patient_index": idx,
                "success": True,
                "prediction": death_prediction,
                "probability": float(death_probability),
                "risk_level": get_risk_level(death_probability),
                "topFactors": explanations[:5]
            })
        
        return jsonify({
            "success": True,
            "total_patients": len(patients),
            "results": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/feature_info', methods=['GET'])
def feature_info():
    """Get information about model features"""
    feature_info = {
        feature: {
            "description": FEATURE_DESCRIPTIONS.get(feature, ""),
            "type": "binary" if feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex'] else "continuous"
        }
        for feature in FEATURE_NAMES
    }
    
    return jsonify({
        "features": feature_info,
        "total_features": len(FEATURE_NAMES)
    })

# ============================================
# MOCK PREDICTION (for testing without model)
# ============================================

def generate_mock_prediction(patient_data):
    """Generate mock prediction when model is not loaded"""
    # Simple rule-based scoring
    risk_score = 0
    
    # Age factor
    if patient_data['age'] > 65:
        risk_score += 0.2
    
    # Ejection fraction (lower is worse)
    if patient_data['ejection_fraction'] < 30:
        risk_score += 0.3
    elif patient_data['ejection_fraction'] < 40:
        risk_score += 0.15
    
    # Serum creatinine (higher is worse)
    if patient_data['serum_creatinine'] > 1.5:
        risk_score += 0.25
    
    # Serum sodium (lower is worse)
    if patient_data['serum_sodium'] < 135:
        risk_score += 0.15
    
    # Comorbidities
    if patient_data['anaemia'] == 1:
        risk_score += 0.1
    if patient_data['high_blood_pressure'] == 1:
        risk_score += 0.08
    if patient_data['diabetes'] == 1:
        risk_score += 0.07
    
    # Add some randomness
    risk_score = min(0.95, max(0.05, risk_score + (np.random.random() * 0.1 - 0.05)))
    
    # Generate mock SHAP values
    mock_shap_values = []
    for feature in FEATURE_NAMES:
        if feature == 'ejection_fraction':
            shap_val = -0.5 if patient_data[feature] > 40 else 0.6
        elif feature == 'serum_creatinine':
            shap_val = 0.3 if patient_data[feature] > 1.5 else -0.2
        elif feature == 'age':
            shap_val = 0.25 if patient_data[feature] > 65 else -0.15
        elif feature == 'serum_sodium':
            shap_val = 0.18 if patient_data[feature] < 135 else -0.12
        else:
            shap_val = np.random.uniform(-0.15, 0.15)
        
        mock_shap_values.append(shap_val)
    
    explanations = generate_shap_explanations(patient_data, mock_shap_values)
    
    return risk_score, explanations

# ============================================
# MODEL SAVING UTILITY (Run after training)
# ============================================

def save_models(calibrated_model, best_model, scaler):
    """
    Call this function after training to save models
    
    Usage:
    save_models(calibrated_model, best_model, scaler)
    """
    import pickle
    
    # Save calibrated model (for predictions)
    with open('model.pkl', 'wb') as f:
        pickle.dump(calibrated_model, f)
    
    # Save XGBoost model (for SHAP)
    xgb_model = best_model.named_steps["xgb"]
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Models saved successfully!")
    print("Files created: model.pkl, xgb_model.pkl, scaler.pkl")

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Heart Failure Prediction API")
    print("=" * 60)
    print(f"Model Status: {'✅ Loaded' if MODEL_LOADED else '⚠️ Mock Mode'}")
    print("Endpoints:")
    print("  - GET  /api/health")
    print("  - POST /api/predict")
    print("  - POST /api/batch_predict")
    print("  - GET  /api/feature_info")
    print("=" * 60)
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)