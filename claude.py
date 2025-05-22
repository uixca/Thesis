import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

FILE_PATH     = '/home3/s3901734/Documents/Thesis/MW.txt'

class EMDDecomposition:
    """Empirical Mode Decomposition implementation"""
    
    def __init__(self, max_imfs=10):
        self.max_imfs = max_imfs
    
    def is_imf(self, signal_data, threshold=0.05):
        """Check if signal satisfies IMF conditions"""
        # Find local maxima and minima
        peaks = signal.find_peaks(signal_data)[0]
        troughs = signal.find_peaks(-signal_data)[0]
        
        # Check condition 1: number of extrema and zero crossings differ by at most 1
        extrema_count = len(peaks) + len(troughs)
        zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
        
        if abs(extrema_count - zero_crossings) > 1:
            return False
        
        # Check condition 2: mean envelope should be close to zero
        if len(peaks) < 2 or len(troughs) < 2:
            return True
        
        # Create envelopes
        upper_env = np.interp(range(len(signal_data)), peaks, signal_data[peaks])
        lower_env = np.interp(range(len(signal_data)), troughs, signal_data[troughs])
        mean_env = (upper_env + lower_env) / 2
        
        return np.mean(np.abs(mean_env)) < threshold
    
    def extract_imf(self, signal_data, max_iterations=100):
        """Extract single IMF from signal"""
        h = signal_data.copy()
        
        for _ in range(max_iterations):
            # Find local maxima and minima
            peaks = signal.find_peaks(h)[0]
            troughs = signal.find_peaks(-h)[0]
            
            if len(peaks) < 2 or len(troughs) < 2:
                break
            
            # Create upper and lower envelopes using cubic spline interpolation
            try:
                upper_env = np.interp(range(len(h)), peaks, h[peaks])
                lower_env = np.interp(range(len(h)), troughs, h[troughs])
                
                # Calculate mean envelope
                mean_env = (upper_env + lower_env) / 2
                
                # Extract component
                h_new = h - mean_env
                
                # Check if it's an IMF
                if self.is_imf(h_new):
                    return h_new
                
                h = h_new
            except:
                break
        
        return h
    
    def decompose(self, signal_data):
        """Decompose signal into IMFs"""
        imfs = []
        residue = signal_data.copy()
        
        for i in range(self.max_imfs):
            imf = self.extract_imf(residue)
            imfs.append(imf)
            residue = residue - imf
            
            # Stop if residue is monotonic or too small
            if len(signal.find_peaks(residue)[0]) < 2 and len(signal.find_peaks(-residue)[0]) < 2:
                break
            
            if np.std(residue) < 0.01:
                break
        
        return np.array(imfs), residue

class HilbertHuangTransform:
    """Hilbert-Huang Transform for extracting IA, IP, IF"""
    
    def __init__(self):
        pass
    
    def calculate_instantaneous_attributes(self, imf):
        """Calculate Instantaneous Amplitude, Phase, and Frequency"""
        # Apply Hilbert transform
        analytic_signal = hilbert(imf)
        
        # Instantaneous Amplitude
        instantaneous_amplitude = np.abs(analytic_signal)
        
        # Instantaneous Phase
        instantaneous_phase = np.angle(analytic_signal)
        
        # Instantaneous Frequency
        instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase)) / (2 * np.pi)
        # Pad to maintain same length
        instantaneous_frequency = np.append(instantaneous_frequency, instantaneous_frequency[-1])
        
        return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

class EEGPreprocessor:
    """EEG Signal Preprocessing Pipeline"""
    
    def __init__(self, sampling_rate=128):
        self.sampling_rate = sampling_rate
        self.emd = EMDDecomposition()
        self.hht = HilbertHuangTransform()
    
    def butterworth_filter(self, data, low_freq, high_freq, order=5):
        """Apply Butterworth bandpass filter"""
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if low <= 0:
            # High-pass filter
            b, a = butter(order, high, btype='low')
        elif high >= 1:
            # Low-pass filter
            b, a = butter(order, low, btype='high')
        else:
            # Band-pass filter
            b, a = butter(order, [low, high], btype='band')
        
        return filtfilt(b, a, data)
    
    def extract_frequency_bands(self, data):
        """Extract six frequency bands as mentioned in paper"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta_low': (12, 16),
            'beta_high': (16, 24),
            'gamma': (24, 40)
        }
        
        filtered_bands = {}
        for band_name, (low, high) in bands.items():
            filtered_bands[band_name] = self.butterworth_filter(data, low, high)
        
        return filtered_bands
    
    def preprocess_channel(self, channel_data):
        """Preprocess single EEG channel"""
        # Apply low-pass filter for denoising (45 Hz cutoff)
        denoised = self.butterworth_filter(channel_data, 0, 45)
        
        # Extract frequency bands
        bands = self.extract_frequency_bands(denoised)
        
        features = []
        
        for band_name, band_data in bands.items():
            # Apply EMD to extract IMFs
            imfs, residue = self.emd.decompose(band_data)
            
            # Apply HHT to each IMF
            for imf in imfs:
                ia, ip, if_freq = self.hht.calculate_instantaneous_attributes(imf)
                
                # Combine IA, IP, IF as features
                combined_features = np.concatenate([ia, ip, if_freq])
                features.append(combined_features)
        
        return np.array(features).flatten()

class BrainDigiCNN:
    """1D CNN model for EEG digit classification"""
    
    def __init__(self, input_shape, num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def build_model(self):
        """Build the 1D CNN architecture as described in paper"""
        model = Sequential([
            # First Convolutional Block
            Conv1D(filters=256, kernel_size=7, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Second Convolutional Block
            Conv1D(filters=128, kernel_size=7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Third Convolutional Block
            Conv1D(filters=64, kernel_size=7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Fourth Convolutional Block
            Conv1D(filters=32, kernel_size=7, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Flatten layer
            Flatten(),
            
            # Fully Connected Layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X, y):
        """Prepare data for training"""
        # Normalize features
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, self.num_classes)
        
        return X_scaled, y_categorical
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model"""
        # Prepare data
        X_train_prep, y_train_prep = self.prepare_data(X_train, y_train)
        X_val_prep, y_val_prep = self.prepare_data(X_val, y_val)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            X_train_prep, y_train_prep,
            validation_data=(X_val_prep, y_val_prep),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape)
        
        predictions = self.model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return self.label_encoder.inverse_transform(predicted_classes)

def load_eeg_data_from_txt(file_path):
    """
    Load EEG data from text file
    Expected format: each row contains [channel1, channel2, ..., channel14, label]
    """
    try:
        # Try to load as CSV first
        data = pd.read_csv(file_path, header=None)
        
        # Assume last column is label, rest are EEG channels
        eeg_data = data.iloc[:, :-1].values
        labels = data.iloc[:, -1].values
        
        print(f"Loaded EEG data shape: {eeg_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
        return eeg_data, labels
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your text file has the format:")
        print("channel1,channel2,...,channel14,digit_label")
        return None, None

def generate_sample_data():
    """Generate sample EEG data for demonstration"""
    np.random.seed(42)
    
    # Simulate EEG data for 10 digits, 100 samples each, 14 channels, 256 time points
    n_samples = 1000
    n_channels = 14
    n_timepoints = 256
    n_digits = 10
    
    X = []
    y = []
    
    for digit in range(n_digits):
        for sample in range(n_samples // n_digits):
            # Generate synthetic EEG-like signal
            eeg_sample = np.random.randn(n_channels, n_timepoints) * 0.1
            
            # Add some digit-specific patterns
            for ch in range(n_channels):
                t = np.linspace(0, 2, n_timepoints)
                # Add frequency components based on digit
                eeg_sample[ch] += 0.5 * np.sin(2 * np.pi * (digit + 1) * t) + \
                                 0.3 * np.sin(2 * np.pi * (digit + 1) * 2 * t)
            
            X.append(eeg_sample)
            y.append(digit)
    
    return np.array(X), np.array(y)

def main():
    """Main execution function"""
    print("EEG Digit Classification - BrainDigiCNN Implementation")
    print("=" * 60)
    
    # Load data (replace with your actual file path)
    file_path = "EP1.01.txt"  # Update with your file path
    eeg_data, labels = load_eeg_data_from_txt(file_path)
    
    # If file not found, generate sample data
    if eeg_data is None:
        print("File not found. Generating sample data for demonstration...")
        eeg_data, labels = generate_sample_data()
    
    print(f"Data shape: {eeg_data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(sampling_rate=128)
    
    # Preprocess data (this might take some time)
    print("Preprocessing EEG data...")
    features = []
    
    for i, sample in enumerate(eeg_data):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(eeg_data)}")
        
        sample_features = []
        for channel in range(sample.shape[0]):
            channel_features = preprocessor.preprocess_channel(sample[channel])
            sample_features.extend(channel_features)
        
        features.append(sample_features)
    
    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")
    
    # Prepare data for CNN
    # Reshape features for 1D CNN input
    max_length = min(5000, features.shape[1])  # Limit feature length for demonstration
    features_reshaped = features[:, :max_length]
    
    if len(features_reshaped.shape) == 2:
        features_reshaped = features_reshaped.reshape(features_reshaped.shape[0], features_reshaped.shape[1], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_reshaped, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Build and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = BrainDigiCNN(input_shape=input_shape, num_classes=len(np.unique(labels)))
    
    print("Building model...")
    model.build_model()
    print(model.model.summary())
    
    print("Training model...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(labels)))
    plt.xticks(tick_marks, np.unique(labels))
    plt.yticks(tick_marks, np.unique(labels))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
