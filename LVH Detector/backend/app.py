from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "x3d_lvh.onnx"  # Update this path
NUM_FRAMES = 16
FRAME_SIZE = 224
LVH_THRESHOLD = 0.37

# Normalization parameters (from training)
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

# ============================================================
# LOAD ONNX MODEL
# ============================================================
print("Loading ONNX model...")
ort_session = ort.InferenceSession(MODEL_PATH)
print("✅ Model loaded successfully")

# ============================================================
# VIDEO PREPROCESSING FUNCTION
# ============================================================
def preprocess_video(video_path, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
    """
    Extract and preprocess frames from video for model inference
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("Empty or unreadable video file")
    
    # Uniform temporal sampling
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled_frames = [frames[i] for i in indices]
    
    # Resize and normalize each frame
    processed_frames = []
    for frame in sampled_frames:
        # Resize
        frame = cv2.resize(frame, (frame_size, frame_size))
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        # Apply normalization
        frame = (frame - MEAN) / STD
        processed_frames.append(frame)
    
    # Stack frames: (T, H, W, C) -> (C, T, H, W)
    video_tensor = np.stack(processed_frames)  # (T, H, W, C)
    video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))  # (C, T, H, W)
    
    # Add batch dimension: (1, C, T, H, W)
    video_tensor = np.expand_dims(video_tensor, axis=0)
    
    return video_tensor.astype(np.float32)

# ============================================================
# API ENDPOINTS
# ============================================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'threshold': LVH_THRESHOLD
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts video file and returns LVH prediction
    """
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Preprocess video
            video_tensor = preprocess_video(tmp_path)
            
            # Run inference
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: video_tensor})
            logits = outputs[0][0]  # Shape: (2,)
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            lvh_probability = float(probs[1])
            prediction = int(lvh_probability >= LVH_THRESHOLD)
            
            # Prepare response
            response = {
                'success': True,
                'lvh_probability': round(lvh_probability, 4),
                'prediction': prediction,
                'prediction_label': 'LVH Detected' if prediction == 1 else 'No LVH',
                'threshold': LVH_THRESHOLD,
                'confidence': round(max(probs) * 100, 2)
            }
            
            return jsonify(response)
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/update-threshold', methods=['POST'])
def update_threshold():
    """
    Update the classification threshold
    """
    global LVH_THRESHOLD
    try:
        data = request.get_json()
        new_threshold = float(data.get('threshold', LVH_THRESHOLD))
        
        if not 0 <= new_threshold <= 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        LVH_THRESHOLD = new_threshold
        return jsonify({
            'success': True,
            'threshold': LVH_THRESHOLD
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)