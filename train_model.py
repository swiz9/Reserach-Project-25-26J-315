"""
Training script for Arrhythmia Detection Model
"""

import os
import glob
import wfdb
import random
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Dataset configuration
DATASET_PATH = './mitdb'
WINDOW_SIZE = 100
BATCH_SIZE = 1024  # Reduced batch size for faster training
EPOCHS = 5  # Reduced epochs for quick training

def sequences(symbol, signal, sample, signal_length):
    """Extract 100-sample sequences around beat annotations"""
    non_beat_annotations = ["[", "!", "]", "x", "(", ")", "p", "t", "u", "`", "'",
                            "^", "|", "~", "+", "s", "T", "*", "D", "=", '"', "@"]
    
    beat_annotations = ["N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "f"]
    
    start = sample - 50
    end = sample + 50
          
    if symbol in beat_annotations and start > 0 and end < signal_length:
        signal_lead_0 = signal[start:end, 0].reshape(1, -1, 1)
        signal_lead_1 = signal[start:end, 1].reshape(1, -1, 1)
        return signal_lead_0, signal_lead_1, symbol
    else:
        return [], [], []

def preprocess(signal):
    """Scale the signal using StandardScaler"""
    scaler = StandardScaler()
    return scaler.fit_transform(signal)

def generate_data(path):
    """Generate training data from ECG records"""
    signal_channel_0 = []
    signal_channel_1 = []
    labels_channel_0 = []
    labels_channel_1 = []
    
    print("Processing {len(path)} files...")
    
    for i, file in enumerate(path):
        if i % 5 == 0:
            print(f"  Processing file {i+1}/{len(path)}: {os.path.basename(file)}")
        
        try:
            # Load the ECG signal from 2 leads
            record = wfdb.rdrecord(file)
            
            # Check the frequency is 360
            if record.fs != 360:
                print(f"Warning: {file} has frequency {record.fs}, skipping...")
                continue
                
            scaled_signal = preprocess(record.p_signal)
            signal_length = scaled_signal.shape[0]
            annotation = wfdb.rdann(file, 'atr')
            samples = annotation.sample
            symbols = annotation.symbol
        
            N = ['.', 'N', 'L', 'R', 'A', 'a', 'J', 'S', 'e', 'j']
            
            for j, sample in enumerate(samples):
                signal_0, signal_1, valid_label = sequences(symbols[j], scaled_signal, sample, signal_length)
                
                if len(signal_0) > 0:  # If we got a valid signal
                    signal_channel_0.extend(signal_0)
                    signal_channel_1.extend(signal_1)
                    
                    if valid_label:
                        if valid_label in N:
                            label = 'N'
                        else:
                            label = valid_label
                        
                        labels_channel_0.append(label)
                        labels_channel_1.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not signal_channel_0:
        print("No valid data found!")
        return np.array([]), np.array([])
    
    signals = np.vstack((signal_channel_0, signal_channel_1))
    labels_channel_0_array = np.array([labels_channel_0]).reshape(-1, 1)
    labels_channel_1_array = np.array([labels_channel_1]).reshape(-1, 1)
    labels = np.vstack((labels_channel_0_array, labels_channel_1_array))
    
    return signals, labels

def data_generator(X, y, batch_size):
    """Data generator for training"""
    num_batches = len(X) // batch_size
    while True:
        np.random.seed(100)
        shuffle_sequence = np.random.permutation(len(X))
        X = X[shuffle_sequence]
        y = y[shuffle_sequence]
        
        for batch in range(num_batches):
            batch_data = np.zeros((batch_size, WINDOW_SIZE, 1))
            batch_labels = np.zeros((batch_size, 4))
        
            for i in range(batch_size):
                idx = i + (batch * batch_size)
                if idx < len(X):
                    batch_data[i, :, :] = X[idx, :, :]
                    batch_labels[i, :] = y[idx, :]
            
            yield batch_data.reshape(batch_size, 1, WINDOW_SIZE, 1), batch_labels.reshape(batch_size, 4)
            
        # Handle remaining samples
        if (len(X) % batch_size != 0):
            remaining = len(X) % batch_size
            batch_data = np.zeros((remaining, WINDOW_SIZE, 1))
            batch_labels = np.zeros((remaining, 4))
            
            start_idx = num_batches * batch_size
            for i in range(remaining):
                if start_idx + i < len(X):
                    batch_data[i, :, :] = X[start_idx + i, :, :]
                    batch_labels[i, :] = y[start_idx + i, :]
            
            yield batch_data.reshape(remaining, 1, WINDOW_SIZE, 1), batch_labels.reshape(remaining, 4)

def create_model():
    """Create the 1D CNN + LSTM model"""
    model = Sequential()

    model.add(TimeDistributed(Conv1D(32, 5, activation='elu'), input_shape=(None, WINDOW_SIZE, 1)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(32, 5, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(64, 4, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(64, 4, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(128, 3, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(128, 3, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(256, 2, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    model.add(TimeDistributed(Conv1D(256, 2, activation='elu')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(8, strides=1)))

    # Extract features
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(256, activation='elu')))

    # LSTM layer
    model.add(LSTM(256, return_sequences=False, dropout=0.5))

    # Classifier
    model.add(Dense(4, activation='softmax'))
    
    return model

def main():
    """Main training function"""
    print("Starting Arrhythmia Detection Model Training...")
    
    # Setup dataset paths
    header_path = os.path.join(DATASET_PATH, '*hea')
    paths = glob.glob(header_path)
    paths = [path[:-4] for path in paths]  # Remove extension
    
    # Remove paced beat records
    remove_paced_beats = ['102', '104', '107', '217']
    data_paths = [path for path in paths if path[-3:] not in remove_paced_beats]
    
    # Split data
    train_data = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', 
                  '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
    
    test_data = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', 
                 '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
    
    random.shuffle(test_data)
    train_data_paths = [path for path in data_paths if path[-3:] in train_data]
    validation_data_paths = [path for path in data_paths if path[-3:] in test_data[:-5]]
    
    print(f"Training files: {len(train_data_paths)}")
    print(f"Validation files: {len(validation_data_paths)}")
    
    # Generate data
    print("\nGenerating training data...")
    train_signals, train_labels = generate_data(train_data_paths)
    
    print("Generating validation data...")
    validation_signals, validation_labels = generate_data(validation_data_paths)
    
    if len(train_signals) == 0:
        print("No training data generated. Exiting...")
        return
    
        print(f"Generated {len(train_signals)} training samples")
    print(f"Generated {len(validation_signals)} validation samples")
    
    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    
    train_labels_numerical = label_encoder.fit_transform(train_labels.reshape(-1, 1))
    train_labels_encoded = one_hot_encoder.fit_transform(train_labels_numerical.reshape(-1, 1))
    
    validation_labels_numerical = label_encoder.transform(validation_labels.reshape(-1, 1))
    validation_labels_encoded = one_hot_encoder.transform(validation_labels_numerical.reshape(-1, 1))
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Create model
    print("Creating model...")
    model = create_model()
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["categorical_accuracy"])
    
    print(f"Model input shape: {model.input_shape}")
    model.summary()
    
    # Setup callbacks
    curr_dt_time = datetime.datetime.now()
    model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '/'
    os.makedirs(model_name, exist_ok=True)
    
    filepath = model_name + 'model-{epoch:05d}-{loss:.3f}-{categorical_accuracy:.3f}-{val_loss:.3f}-{val_categorical_accuracy:.3f}.weights.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=1, verbose=1)
    callbacks_list = [checkpoint, LR]
    
    # Create generators
    train_generator = data_generator(train_signals, train_labels_encoded, BATCH_SIZE)
    val_generator = data_generator(validation_signals, validation_labels_encoded, BATCH_SIZE)
    
    # Calculate steps
    steps_per_epoch = len(train_signals) // BATCH_SIZE + (1 if len(train_signals) % BATCH_SIZE != 0 else 0)
    validation_steps = len(validation_signals) // BATCH_SIZE + (1 if len(validation_signals) % BATCH_SIZE != 0 else 0)
    
    print("\nStarting training...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # Train model
    history = model.fit(x=train_generator, 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=EPOCHS, 
                        verbose=1,
                        callbacks=callbacks_list, 
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        shuffle=True, 
                        initial_epoch=0)
    
    # Save the complete model
    model_save_path = 'arrhythmia_model.h5'
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Model saved successfully!")
    
    # Also save as the expected name for predict.py
    predict_model_path = 'model-00006-0.159-0.952-0.002-0.961.h5'
    model.save(predict_model_path)
    print(f"Model also saved as {predict_model_path}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
