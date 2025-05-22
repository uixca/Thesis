#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TF logs

import sys, time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mne
from scipy.signal import hilbert
from PyEMD import EMD
from tensorflow.keras import Input, layers, models
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configuration
# ----------------------------
FILE_PATH     = '/home3/s3901734/Documents/Thesis/MW.txt'
TARGET_LENGTH = 256
SFREQ         = 128
F_BANDS       = [(0.5,4), (4,8), (8,12), (12,16), (16,24), (24,40)]
N_IMFS        = 10
N_CLASSES     = 10
BATCH_SIZE    = 32
EPOCHS        = 10
PATIENCE      = 10
N_JOBS        = max(1, multiprocessing.cpu_count() - 1)
USE_SAVED_FEATURES = True
SAVE_DIR      = 'features'
MODEL_PATH    = 'best_model.h5'


# ----------------------------
# (0) Debug helpers
# ----------------------------
def debug_data_stats(X, y, tag=""):
    print(f"\n=== DATA STATS {tag} ===")
    print("X shape:", X.shape)
    print(" X min, max, mean, std:", X.min(), X.max(), X.mean(), X.std())
    print(" y distribution:", np.bincount(y))
    print("=" * 30)

def test_overfit_cnn(X, y):
    print("\n--- OVERFIT TEST (100 samples) ---")
    small_X = X[:100]
    small_y = y[:100]
    model = build_brain_digi_cnn((TARGET_LENGTH, small_X.shape[2]))
    hist = model.fit(
        small_X, to_categorical(small_y, N_CLASSES),
        epochs=50, batch_size=8, verbose=2
    )
    acc = hist.history['accuracy'][-1]
    print(f"Final training accuracy on 100 samples: {acc:.3f}")
    print("-" * 30)


def test_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n--- LOGISTIC REGRESSION BASELINE ---")
    # Flatten features
    Xtr = X_train.reshape(len(X_train), -1)
    Xvl = X_val.reshape(len(X_val), -1)
    # Use a subset if too big
    n_tr = min(len(Xtr), 2000)
    n_vl = min(len(Xvl), 1000)
    clf = LogisticRegression(max_iter=1000).fit(Xtr[:n_tr], y_train[:n_tr])
    score = clf.score(Xvl[:n_vl], y_val[:n_vl])
    print(f"LR val accuracy ({n_vl} samples): {score:.3f}")
    print("-" * 30)


# ----------------------------
# (1–5) your existing functions
# ----------------------------
def load_mbd_meta(path):
    # … identical to your current code …
    df = pd.read_csv(path, sep='\t', header=None, comment='#',
                     usecols=[0,1,3,4,6],
                     names=['id','event_id','channel','code','signal'],
                     dtype={'id':int,'event_id':int,'code':int})
    df = df[df['code'] >= 0]
    return df

def build_raws(df):
    # … identical to your current code …
    raws, labels = [], []
    for (eid, lbl), grp in df.groupby(['event_id','code']):
        grp = grp.sort_values('channel')
        chans = [np.fromstring(s, sep=',') for s in grp['signal']]
        data = np.stack([np.pad(c, (0, max(0, TARGET_LENGTH-len(c))), 'constant')[:TARGET_LENGTH]
                         for c in chans])
        info = mne.create_info(ch_names=[f"Ch{i}" for i in range(data.shape[0])],
                               sfreq=SFREQ, ch_types='eeg')
        raws.append(mne.io.RawArray(data, info, verbose=False))
        labels.append(lbl)
    return raws, np.array(labels)

def bandpass_data(raw):
    filtered = []
    for l, h in F_BANDS:
        filtered.append(
            raw.copy().filter(l, h, method='iir',
                              iir_params={'order':5,'ftype':'butter'},
                              verbose=False).get_data()
        )
    return np.stack(filtered)

def extract_imfs(sig):
    # … identical …

    if len(sig)==0 or np.isnan(sig).any():
        return np.zeros((N_IMFS, len(sig) if len(sig)>0 else TARGET_LENGTH))
    sig = (sig - sig.mean()) / (sig.std() if sig.std()>0 else 1.)
    imfs = EMD().emd(sig, max_imf=N_IMFS)
    if len(imfs)==0:
        return np.zeros((N_IMFS, sig.shape[0]))
    if len(imfs)<N_IMFS:
        last = imfs[-1]
        pad  = np.vstack([last]*(N_IMFS-len(imfs)))
        imfs = np.vstack([imfs, pad])
    return imfs[:N_IMFS]

def hht_features(imfs):
    feats = []
    for imf in imfs:
        a = hilbert(imf)
        amp  = np.abs(a)
        ph   = np.angle(a)
        inst = np.diff(ph, prepend=ph[0])
        feats.extend([amp, ph, inst])
    return np.stack(feats)

def process_single_trial(raw, label):
    bd = bandpass_data(raw)            # (6, ch, 256)
    tf = []
    for band in bd:
        for ch in band:
            tf.append(hht_features(extract_imfs(ch)))
    return np.vstack(tf).T, label      # (256, 6*ch*30)

class EEGSequence(Sequence):
    def __init__(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.indices)/self.batch_size))
    def __getitem__(self, idx):
        idxs = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        Xb = np.array([self.X[i] for i in idxs])
        yb = to_categorical(self.y[idxs], N_CLASSES)
        return Xb, yb
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_brain_digi_cnn(input_shape):
    inp = Input(shape=input_shape)
    x = layers.Conv1D(256,7,'relu','same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128,5,'relu','same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64,5,'relu','same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(32,3,'relu','same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,'relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64,'relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES,'softmax')(x)
    m = models.Model(inp, out)
    m.compile('adam','categorical_crossentropy', ['accuracy'])
    return m

def compute_features_for_all(raws, labels):
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_trial)(r,l) for r,l in zip(raws,labels)
    )
    X, y = zip(*results)
    X = np.stack(X); y = np.array(y)
    np.save(os.path.join(SAVE_DIR,'X.npy'), X)
    np.save(os.path.join(SAVE_DIR,'y.npy'), y)
    return X, y

def plot_training_history(history):
    # … identical …

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend(); plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend(); plt.title('Loss')
    plt.tight_layout()
    plt.savefig('training_history.png')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"Digit {i}" for i in range(N_CLASSES)])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(N_CLASSES):
        TP = cm[i,i]
        FP = cm[:,i].sum()-TP
        FN = cm[i,:].sum()-TP
        TN = cm.sum() - (TP+FP+FN)
        specs.append(TN/(TN+FP) if (TN+FP)>0 else 0.)
    print("Specificities:", np.round(specs,4))
    return specs


# ----------------------------
# (6) Main
# ----------------------------
if __name__ == "__main__":
    # 1) Load or compute features
    if USE_SAVED_FEATURES and os.path.exists(os.path.join(SAVE_DIR,'X.npy')):
        print("Loading precomputed features…")
        X = np.load(os.path.join(SAVE_DIR,'X.npy'))
        y = np.load(os.path.join(SAVE_DIR,'y.npy'))
    else:
        df = load_mbd_meta(FILE_PATH)
        raws, labels = build_raws(df)
        X, y = compute_features_for_all(raws, labels)
        del raws; gc.collect()

    # ---- DEBUG STEP 1 ----
    debug_data_stats(X, y, tag="(after load)")

    # 2) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # ---- DEBUG STEP 2 ----
    debug_data_stats(X_train, y_train, tag="(train)")
    debug_data_stats(X_val,   y_val,   tag="(val)")

    # 3) Test tiny overfit
    test_overfit_cnn(X_train, y_train)

    # 4) Logistic regression baseline
    test_logistic_regression(X_train, y_train, X_val, y_val)

    # 5) Build data loaders & model as before
    train_seq = EEGSequence(X_train, y_train)
    val_seq   = EEGSequence(X_val,   y_val, shuffle=False)
    model     = build_brain_digi_cnn((TARGET_LENGTH, X_train.shape[2]))

    callbacks = [
        EarlyStopping('val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(MODEL_PATH, 'val_accuracy', save_best_only=True, verbose=1)
    ]

    # 6) Train
    history = model.fit(
        train_seq, validation_data=val_seq,
        epochs=EPOCHS, callbacks=callbacks, verbose=2
    )

    # 7) Evaluate
    plot_training_history(history)
    y_pred = np.argmax(model.predict(val_seq), axis=1)
    plot_confusion_matrix(y_val, y_pred)
    print(classification_report(y_val, y_pred, digits=4,
          target_names=[f"Digit {i}" for i in range(N_CLASSES)]))
    compute_specificity(y_val, y_pred)

    print("\nDone!")
