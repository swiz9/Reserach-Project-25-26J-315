from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('best_extratrees_pca.pkl')

# Global variables for preprocessing (these should match your training)
FEATURE_NAMES = ['gender', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
                 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
                 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

# Initialize scaler and PCA (load from saved objects if available)
scaler = StandardScaler()
pca = PCA(n_components=12, random_state=42)

# Load preprocessing objects
try:
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
except:
    print("Warning: Preprocessing objects not found. You need to save them during training.")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in correct order
        features = {
            'gender': data.get('gender', 0),
            'age': data.get('age', 50),
            'education': data.get('education', 2),
            'currentSmoker': data.get('currentSmoker', 0),
            'cigsPerDay': data.get('cigsPerDay', 0),
            'BPMeds': data.get('BPMeds', 0),
            'prevalentStroke': data.get('prevalentStroke', 0),
            'prevalentHyp': data.get('prevalentHyp', 0),
            'diabetes': data.get('diabetes', 0),
            'totChol': data.get('totChol', 200),
            'sysBP': data.get('sysBP', 120),
            'diaBP': data.get('diaBP', 80),
            'BMI': data.get('BMI', 25),
            'heartRate': data.get('heartRate', 70),
            'glucose': data.get('glucose', 85)
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Feature engineering (must match training)
        df['pulse_pressure'] = df['sysBP'] - df['diaBP']
        df['MAP'] = (2 * df['diaBP'] + df['sysBP']) / 3
        df['age_x_sysBP'] = df['age'] * df['sysBP']
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Apply PCA
        X_pca = pca.transform(X_scaled)
        
        # Make prediction
        prediction = model.predict(X_pca)[0]
        probability = model.predict_proba(X_pca)[0]
        
        # Calculate risk level
        risk_prob = probability[1] * 100
        if risk_prob < 15:
            risk_level = "Low"
        elif risk_prob < 30:
            risk_level = "Moderate"
        elif risk_prob < 50:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return jsonify({
            'prediction': int(prediction),
            'probability': {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            },
            'risk_percentage': float(risk_prob),
            'risk_level': risk_level,
            'interpretation': f"{'High' if prediction == 1 else 'Low'} risk of CHD in 10 years"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        
        # Extract and preprocess features (same as predict)
        features = {
            'gender': data.get('gender', 0),
            'age': data.get('age', 50),
            'education': data.get('education', 2),
            'currentSmoker': data.get('currentSmoker', 0),
            'cigsPerDay': data.get('cigsPerDay', 0),
            'BPMeds': data.get('BPMeds', 0),
            'prevalentStroke': data.get('prevalentStroke', 0),
            'prevalentHyp': data.get('prevalentHyp', 0),
            'diabetes': data.get('diabetes', 0),
            'totChol': data.get('totChol', 200),
            'sysBP': data.get('sysBP', 120),
            'diaBP': data.get('diaBP', 80),
            'BMI': data.get('BMI', 25),
            'heartRate': data.get('heartRate', 70),
            'glucose': data.get('glucose', 85)
        }
        
        df = pd.DataFrame([features])
        df['pulse_pressure'] = df['sysBP'] - df['diaBP']
        df['MAP'] = (2 * df['diaBP'] + df['sysBP']) / 3
        df['age_x_sysBP'] = df['age'] * df['sysBP']
        
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)
        
        # Generate SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_pca)
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        feature_names = [f'PC{i+1}' for i in range(12)]
        shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        # Sort by absolute value
        indices = np.argsort(np.abs(shap_vals))[::-1][:6]
        
        plt.barh([feature_names[i] for i in indices], 
                [shap_vals[i] for i in indices])
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title('Top 6 Feature Contributions')
        plt.tight_layout()
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return jsonify({
            'shap_plot': img_str,
            'feature_importance': {
                feature_names[i]: float(shap_vals[i]) 
                for i in indices
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        patients = data.get('patients', [])
        
        results = []
        for patient in patients:
            features = {k: patient.get(k, 0) for k in FEATURE_NAMES}
            df = pd.DataFrame([features])
            
            df['pulse_pressure'] = df['sysBP'] - df['diaBP']
            df['MAP'] = (2 * df['diaBP'] + df['sysBP']) / 3
            df['age_x_sysBP'] = df['age'] * df['sysBP']
            
            X_scaled = scaler.transform(df)
            X_pca = pca.transform(X_scaled)
            
            prediction = model.predict(X_pca)[0]
            probability = model.predict_proba(X_pca)[0]
            
            results.append({
                'patient_id': patient.get('id', ''),
                'prediction': int(prediction),
                'risk_probability': float(probability[1]),
                'risk_level': 'High' if probability[1] > 0.3 else 'Low'
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)