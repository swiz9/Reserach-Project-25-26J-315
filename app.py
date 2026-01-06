"""
Flask Web Application for Arrhythmia Detection
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import wfdb
from zone_classifier import default_zone_policy, get_zone_color

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'txt', 'dat'}

# Class mapping
ACTUAL_CLASS_MAPPING = {
    0: 'E',  # Ventricular escape
    1: 'F',  # Fusion
    2: 'N',  # Normal
    3: 'V'   # Premature ventricular contraction
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_trained_model():
    """Load the trained arrhythmia detection model"""
    try:
        model_path = 'model-00006-0.159-0.952-0.002-0.961.h5'
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_ecg_data(file_path):
    """Process uploaded ECG data file"""
    try:
        # Read CSV file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'MLII' in df.columns:
                ecg_signal = df['MLII'].values
            else:
                ecg_signal = df.iloc[:, 0].values
        else:
            # For other file types, try to read as CSV
            df = pd.read_csv(file_path, header=None)
            ecg_signal = df.iloc[:, 0].values
        
        # Normalize the signal
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        return ecg_signal
    except Exception as e:
        print(f"Error processing ECG data: {e}")
        return None

def create_ecg_plot(ecg_signal, title="ECG Signal"):
    """Create a plot of the ECG signal"""
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_signal, linewidth=1)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_str

def predict_ecg_segments(model, ecg_signal, window_size=100):
    """Predict arrhythmia classes for ECG segments"""
    try:
        # Split signal into windows
        segments = []
        for i in range(0, len(ecg_signal) - window_size, window_size):
            segment = ecg_signal[i:i + window_size]
            segments.append(segment)
        
        if not segments:
            return None, None, None
        
        segments = np.array(segments)
        
        # Reshape for model input: (batch_size, 1, 100, 1)
        segments_reshaped = segments.reshape(-1, 1, window_size, 1)
        
        # Make predictions
        predictions = model.predict(segments_reshaped, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Convert to class labels
        predicted_labels = [ACTUAL_CLASS_MAPPING[idx] for idx in predicted_classes]
        
        return predicted_labels, confidence_scores, segments
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

# Load the model at startup
model = load_trained_model()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process ECG data
        ecg_signal = process_ecg_data(filepath)
        if ecg_signal is None:
            return jsonify({'error': 'Error processing ECG data'}), 400
        
        # Make predictions
        predicted_labels, confidence_scores, segments = predict_ecg_segments(model, ecg_signal)
        if predicted_labels is None:
            return jsonify({'error': 'Error during prediction'}), 400
        
        # Create ECG plot
        ecg_plot = create_ecg_plot(ecg_signal, "Uploaded ECG Signal")
        
        # Calculate overall statistics
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        total_segments = len(predicted_labels)
        
        # Get zone classification for majority class
        majority_label = unique_labels[np.argmax(counts)]
        avg_confidence = np.mean(confidence_scores)
        zone, explanation = default_zone_policy(majority_label, avg_confidence)
        zone_color = get_zone_color(zone)
        
        # Prepare results
        results = {
            'success': True,
            'ecg_plot': ecg_plot,
            'total_segments': total_segments,
            'predictions': [
                {
                    'segment_id': i,
                    'predicted_class': predicted_labels[i],
                    'confidence': float(confidence_scores[i])
                }
                for i in range(min(len(predicted_labels), 20))  # Limit to first 20 for display
            ],
            'statistics': {
                'class_distribution': {
                    label: int(count) for label, count in zip(unique_labels, counts)
                },
                'majority_class': majority_label,
                'average_confidence': float(avg_confidence),
                'zone': zone,
                'zone_color': zone_color,
                'explanation': explanation
            }
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_sample')
def predict_sample():
    """Predict on a sample ECG beat"""
    try:
        # Generate synthetic ECG-like signal
        t = np.linspace(0, 2*np.pi, 100)
        sample_signal = np.sin(t) + 0.3*np.sin(3*t) + 0.1*np.sin(5*t)
        sample_signal = (sample_signal - np.mean(sample_signal)) / np.std(sample_signal)
        
        # Reshape for prediction
        sample_reshaped = sample_signal.reshape(1, 1, 100, 1)
        
        # Make prediction
        prediction = model.predict(sample_reshaped, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        predicted_class = ACTUAL_CLASS_MAPPING[predicted_class_idx]
        
        # Get zone classification
        zone, explanation = default_zone_policy(predicted_class, confidence)
        zone_color = get_zone_color(zone)
        
        # Create plot
        sample_plot = create_ecg_plot(sample_signal, "Sample ECG Beat")
        
        return jsonify({
            'success': True,
            'sample_plot': sample_plot,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'zone': zone,
            'zone_color': zone_color,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing sample: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Arrhythmia Detection Web Application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
