#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TF logs

import sys, time, gc
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import mne
from scipy.signal import hilbert
from PyEMD import EMD
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, layers, models
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import multiprocessing
from joblib import Parallel, delayed
import psutil
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# ----------------------------
# Configuration
# ----------------------------
FILE_PATH     = '/home3/s3901734/Documents/Thesis/EP1.01.txt'
TARGET_LENGTH = 256
SFREQ         = 128
F_BANDS       = [(0.5,4), (4,8), (8,12), (12,16), (16,24), (24,40)]
N_IMFS        = 10
N_CLASSES     = 10
BATCH_SIZE    = 32
EPOCHS        = 15
PATIENCE      = 10
N_JOBS        = 10  # Conservative parallel processing
MODEL_PATH    = 'best_model.h5'

# ----------------------------
# Memory monitoring function
# ----------------------------
def check_memory_usage():
    """Monitor current memory usage."""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    available_gb = memory.available / (1024**3)
    percent = memory.percent
    print(f"Memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available ({percent:.1f}%)", flush=True)
    return used_gb, available_gb, percent

# ----------------------------
# 1) Load metadata
# ----------------------------
def load_mbd_meta(path):
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr, flush=True)
        sys.exit(1)
        
    try:
        print("1) Reading metadata…", flush=True)
        t0 = time.time()
        df = pd.read_csv(
            path,
            sep='\t', header=None, comment='#',
            usecols=[0,1,3,4,6],
            names=['id','event_id','channel','code','signal'],
            dtype={'id':int,'event_id':int,'code':int},
            engine='c', low_memory=False
        )
        print(f"   → read {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"ERROR loading file: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
        
    # Filter out trials with negative codes
    df = df[df['code'] >= 0]
    
    # Check if we have data for all classes
    unique_codes = df['code'].unique()
    if len(unique_codes) < N_CLASSES:
        print(f"WARNING: Only found {len(unique_codes)} classes instead of {N_CLASSES}.", flush=True)
        print(f"   → Classes found: {sorted(unique_codes)}", flush=True)
    
    print(f"   → {df['event_id'].nunique():,} unique trials after filtering", flush=True)
    return df

# ----------------------------
# 2) Build RawArray epochs
# ----------------------------
def build_raws(df):
    raws, labels = [], []
    event_groups = df.groupby(['event_id','code'])
    total_groups = len(event_groups)
    
    print(f"2) Building {total_groups} RawArray epochs...", flush=True)
    
    for i, ((eid, lbl), grp) in enumerate(event_groups):
        # Show progress
        if i % 100 == 0:
            print(f"   → Processing {i}/{total_groups} trials...", flush=True)
            
        # Sort by channel to ensure consistent order
        grp = grp.sort_values('channel')
        
        # Check if we have the expected number of channels
        if len(grp) == 0:
            print(f"WARNING: Skipping empty trial {eid}", flush=True)
            continue
            
        try:
            # Convert string representations to numpy arrays
            chans = []
            for s in grp['signal']:
                try:
                    chan_data = np.fromstring(s, sep=',')
                    chans.append(chan_data)
                except Exception as e:
                    print(f"ERROR parsing signal data: {e}", flush=True)
                    raise
                    
            data = np.stack(chans)  # (n_channels, raw_len)
            
            # Check if data has expected dimensions
            if data.shape[0] == 0:
                print(f"WARNING: Trial {eid} has no channels. Skipping.", flush=True)
                continue
                
            # Pad/trim to target length
            if data.shape[1] < TARGET_LENGTH:
                pad = TARGET_LENGTH - data.shape[1]
                data = np.pad(data, ((0,0),(0,pad)), 'constant')
            else:
                data = data[:, :TARGET_LENGTH]
                
            # Create MNE Raw object
            info = mne.create_info(
                ch_names=[f"Ch{i}" for i in range(data.shape[0])],
                sfreq=SFREQ, ch_types='eeg'
            )
            raw = mne.io.RawArray(data, info, verbose=False)
            
            # Add to lists
            raws.append(raw)
            labels.append(lbl)
            
        except Exception as e:
            print(f"ERROR processing trial {eid}: {e}", flush=True)
            continue
            
    # Clean up memory
    del df; gc.collect()
    
    # Check if we have any valid trials
    if len(raws) == 0:
        print("ERROR: No valid trials found. Check your data.", file=sys.stderr, flush=True)
        sys.exit(1)
        
    print(f"   → Successfully built {len(raws):,} RawArray epochs", flush=True)
    return raws, np.array(labels)

# ----------------------------
# 3) Filtering + EMD + HHT
# ----------------------------
def bandpass_data(raw):
    """Apply bandpass filtering to raw data for each frequency band."""
    filtered_data = []
    for l, h in F_BANDS:
        try:
            filtered = raw.copy().filter(
                l, h, method='iir',
                iir_params={'order':5,'ftype':'butter'},
                verbose=False
            ).get_data()
            filtered_data.append(filtered)
        except Exception as e:
            print(f"ERROR in bandpass filter ({l}-{h} Hz): {e}", flush=True)
            # Return zeros if filtering fails
            filtered_data.append(np.zeros_like(raw.get_data()))
    
    return np.stack(filtered_data)  # (6,14,256)

def extract_imfs(sig):
    """
    Decompose sig into at most N_IMFS intrinsic mode functions.
    If no IMFs are found, return a zero-array of shape (N_IMFS, len(sig)).
    If fewer than N_IMFS, repeat the last IMF to pad up.
    """
    try:
        # Check if input has valid shape and values
        if len(sig) == 0 or np.isnan(sig).any() or np.isinf(sig).any():
            return np.zeros((N_IMFS, len(sig) if len(sig) > 0 else TARGET_LENGTH))
        
        # Normalize signal to improve EMD stability
        sig = (sig - np.mean(sig)) / (np.std(sig) if np.std(sig) > 0 else 1.0)
        
        # Apply EMD with a timeout
        imfs = EMD().emd(sig, max_imf=N_IMFS)
        n_samples = sig.shape[0]

        # 1) No IMFs found at all → zeros
        if len(imfs) == 0:
            return np.zeros((N_IMFS, n_samples))

        # 2) Too few IMFs → pad by repeating last
        if len(imfs) < N_IMFS:
            last = imfs[-1]
            pad = np.vstack([last] * (N_IMFS - len(imfs)))
            return np.vstack([imfs, pad])

        # 3) Too many → truncate
        return imfs[:N_IMFS]
        
    except Exception as e:
        print(f"ERROR in EMD extraction: {e}", flush=True)
        return np.zeros((N_IMFS, len(sig)))

def hht_features(imfs):
    """Extract amplitude, phase, and instantaneous frequency from IMFs using the Hilbert transform."""
    feats = []
    for imf in imfs:
        try:
            # Apply Hilbert transform
            a = hilbert(imf)
            
            # Calculate amplitude (envelope)
            amplitude = np.abs(a)
            
            # Calculate phase
            phase = np.angle(a)
            
            # Calculate instantaneous frequency (derivative of phase)
            inst_freq = np.diff(phase, prepend=phase[0])
            
            # Add features
            feats.extend([amplitude, phase, inst_freq])
            
        except Exception as e:
            print(f"ERROR in Hilbert transform: {e}", flush=True)
            # Return zeros if Hilbert transform fails
            zeros = np.zeros_like(imf)
            feats.extend([zeros, zeros, zeros])
    
    return np.stack(feats)  # (30,256)

def process_single_trial(raw, label):
    """Process a single trial to extract features."""
    try:
        # Apply bandpass filtering
        bd = bandpass_data(raw)  # (6,14,256)
        
        # Extract features for each band and channel
        tf = []
        for band in bd:
            for ch in band:
                tf.append(hht_features(extract_imfs(ch)))
                
        # Stack features
        features = np.vstack(tf).T  # (256, 6*14*30)
        return features, label
        
    except Exception as e:
        print(f"ERROR processing trial: {e}", flush=True)
        # Return zeros if processing fails
        return np.zeros((TARGET_LENGTH, 6*14*30)), label

# ----------------------------
# 4) Memory-efficient data splitting
# ----------------------------
def manual_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Memory-efficient manual train/test split that doesn't create unnecessary copies.
    """
    print("4) Manual memory-efficient data splitting...", flush=True)
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get stratified indices for each class
    unique_classes = np.unique(y)
    train_indices = []
    val_indices = []
    
    for class_label in unique_classes:
        class_mask = (y == class_label)
        class_indices = np.where(class_mask)[0]
        
        # Shuffle indices for this class
        np.random.shuffle(class_indices)
        
        # Split this class
        n_val = int(len(class_indices) * test_size)
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    print(f"   → Train indices: {len(train_indices)}, Val indices: {len(val_indices)}", flush=True)
    
    # Create train arrays by indexing (more memory efficient than copying)
    print("   → Creating training arrays...", flush=True)
    X_train = X[train_indices]  # This creates a copy, but we do it one at a time
    y_train = y[train_indices]
    
    print("   → Creating validation arrays...", flush=True)
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    # Immediately delete original arrays to free memory
    print("   → Freeing original arrays...", flush=True)
    del X, y
    gc.collect()
    
    print(f"   → Training set: {X_train.shape[0]} samples", flush=True)
    print(f"   → Validation set: {X_val.shape[0]} samples", flush=True)
    
    return X_train, X_val, y_train, y_val

# ----------------------------
# 5) Keras Sequence for batches
# ----------------------------
class EEGSequence(Sequence):
    def __init__(self, X, y, 
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 **kwargs):
        # Accept and pass through any Keras kwargs
        super().__init__(**kwargs)

        self.X         = X
        self.y         = y
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indices    = np.arange(len(X))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, idx):
        """Get batch at position idx."""
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X_batch = np.array([self.X[i] for i in batch_indices])
        y_batch = to_categorical(np.array([self.y[i] for i in batch_indices]), N_CLASSES)
        return X_batch, y_batch
        
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# ----------------------------
# 6) Build BrainDigiCNN
# ----------------------------
def build_brain_digi_cnn(input_shape):
    """Build a CNN model for EEG classification."""
    inp = Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv1D(256, 7, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout
    
    # Second convolutional block
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout
    
    # Third convolutional block
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout
    
    # Fourth convolutional block
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)  # Add dropout
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Add dropout
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Add dropout
    out = layers.Dense(N_CLASSES, activation='softmax')(x)
    
    # Create and compile model
    model = models.Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def compute_features_for_all(raws, labels):
    """Compute features for all trials (without saving to disk)."""
    print(f"3) Computing features for {len(raws)} trials using {N_JOBS} processes...", flush=True)
    t0 = time.time()
    
    # Process trials in parallel
    results = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(process_single_trial)(raw, label) for raw, label in zip(raws, labels)
    )
    
    # Separate features and labels
    X_list, y_list = zip(*results)
    X = np.stack(X_list)
    y = np.array(y_list)
    
    print(f"   → Computed features in {time.time()-t0:.1f}s", flush=True)
    print(f"   → Features shape: {X.shape}, Labels shape: {y.shape}", flush=True)
    
    return X, y

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png", flush=True)
    
def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"Digit {i}" for i in range(N_CLASSES)])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png", flush=True)
    
def compute_specificity(y_true, y_pred):
    """Compute and print specificity for each class."""
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    
    print("\nSpecificity per class:")
    for i in range(N_CLASSES):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificities.append(spec)
        print(f"  Digit {i}: Specificity = {spec:.4f}")
        
    print(f"Average specificity: {np.mean(specificities):.4f}")
    
    return specificities

# ----------------------------
# 7) Main
# ----------------------------
if __name__ == "__main__":
    try:
        print("Starting EEG processing with memory optimization...", flush=True)
        
        # Initial memory check
        print("Initial memory state:")
        check_memory_usage()
        
        # Load and process data
        df_meta = load_mbd_meta(FILE_PATH)
        raws, labels = build_raws(df_meta)
        
        print("After loading raw data:")
        check_memory_usage()
        
        # Compute features
        X, y = compute_features_for_all(raws, labels)
        
        # Clean up raw data immediately
        del raws
        gc.collect()
        
        print("After feature computation and cleanup:")
        check_memory_usage()
        
        # Use memory-efficient splitting
        X_train, X_val, y_train, y_val = manual_train_test_split(X, y, test_size=0.3, random_state=42)
        
        print("After data splitting:")
        check_memory_usage()
        
        # Create data generators
        train_seq = EEGSequence(X_train, y_train)
        val_seq = EEGSequence(X_val, y_val, shuffle=False)
        
        # Check feature dimension
        feat_dim = X_train.shape[2]
        print(f"   → Feature dimension: {feat_dim}", flush=True)
        
        # Build model
        print("5) Building model...", flush=True)
        model = build_brain_digi_cnn((TARGET_LENGTH, feat_dim))
        model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1
            ),
            ModelCheckpoint(
                MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("6) Training model...", flush=True)
        t0 = time.time()
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2
        )
        print(f"   → Training completed in {time.time()-t0:.1f}s", flush=True)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on validation set
        print("7) Evaluating model...", flush=True)
        y_pred_probs = model.predict(val_seq, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = y_val
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            digits=4,
            target_names=[f"Digit {i}" for i in range(N_CLASSES)]
        ))
        
        # Compute specificity
        specificities = compute_specificity(y_true, y_pred)
        
        print("\nDone! 🎉", flush=True)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
