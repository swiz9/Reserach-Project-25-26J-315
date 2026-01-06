import numpy as np
import pandas as pd
import wfdb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model-00006-0.159-0.952-0.002-0.961.h5"
RECORD_NAME = "100"  # MIT-BIH record name
WINDOW_SIZE = 100

# Class mapping for the model
ACTUAL_CLASS_MAPPING = {
    0: 'E',  # Ventricular escape
    1: 'F',  # Fusion
    2: 'N',  # Normal
    3: 'V'   # Premature ventricular contraction
}

def get_sample_beat(record_name, beat_index=0):
    """Get a sample beat from MIT-BIH record"""
    try:
        # Load the ECG signal
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
        
        # Get the sample at the specified beat index
        if beat_index < len(annotation.sample):
            sample = annotation.sample[beat_index]
            symbol = annotation.symbol[beat_index]
            
            # Extract 100 samples around the beat (50 before, 50 after)
            start = max(0, sample - 50)
            end = min(record.p_signal.shape[0], sample + 50)
            
            if end - start == WINDOW_SIZE:
                # Get the first lead signal
                beat_signal = record.p_signal[start:end, 0]
                
                # Normalize the signal
                scaler = StandardScaler()
                beat_signal = scaler.fit_transform(beat_signal.reshape(-1, 1)).flatten()
                
                return beat_signal, symbol
            else:
                return None, None
        else:
            return None, None
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None, None

def generate_grad_cam(model, signal, class_idx):
    """Generate Grad-CAM heatmap (simplified version)"""
    # This is a placeholder for Grad-CAM implementation
    # In a full implementation, you would compute gradients and create heatmaps
    return np.zeros_like(signal)

print("Loading model from:", MODEL_PATH)
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("Model input shape:", model.input_shape)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure the model file exists and was trained properly.")
    exit(1)

# Step 1: Get sample ECG beat
print("Getting sample ECG beat...")
beat_signal, actual_symbol = get_sample_beat(RECORD_NAME, beat_index=10)

if beat_signal is None:
    print("Could not get sample beat. Trying to generate synthetic data...")
    # Generate synthetic ECG-like signal as fallback
    t = np.linspace(0, 2*np.pi, WINDOW_SIZE)
    beat_signal = np.sin(t) + 0.3*np.sin(3*t) + 0.1*np.sin(5*t)
    beat_signal = (beat_signal - np.mean(beat_signal)) / np.std(beat_signal)
    actual_symbol = "Synthetic"

print(f"Got beat signal of length: {len(beat_signal)}")
print(f"   Actual symbol: {actual_symbol}")

# Step 2: Prepare input for model (reshape to match model input shape)
# Model expects: (batch_size, 1, 100, 1)
segments = beat_signal.reshape(1, 1, WINDOW_SIZE, 1)
print(f"Prepared segments for prediction, shape: {segments.shape}")

# Step 3: Predict
print("Running prediction...")
try:
    preds = model.predict(segments, verbose=0)
    predicted_class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    predicted_class = ACTUAL_CLASS_MAPPING[predicted_class_idx]
    
    print("\nPrediction Results:")
    print(f"   Predicted Class: {predicted_class} (Index: {predicted_class_idx})")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Actual Symbol: {actual_symbol}")
    
    # Generate Grad-CAM (placeholder)
    heatmap = generate_grad_cam(model, beat_signal, predicted_class_idx)
    
    # Save results
    output = pd.DataFrame({
        "beat_index": [0],
        "predicted_class": [predicted_class],
        "predicted_class_idx": [predicted_class_idx],
        "confidence": [confidence],
        "actual_symbol": [actual_symbol]
    })
    output.to_csv("predicted_results.csv", index=False)
    print("Results saved to predicted_results.csv")
    
except Exception as e:
    print(f"Error during prediction: {e}")
    print("This might be due to model architecture mismatch.")
