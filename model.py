# ============================================================
# model.py — TekMedica ECG Classification Kit (Chapman Dataset)
# ============================================================

import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import find_peaks, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import random

# -----------------------------
# Configuration
# -----------------------------
AFIB_DIR = r"D:\AFIB TRAIN\AFIB TRAIN\chapman_afib"
NONAFIB_DIR = r"D:\NONAFIB TRAIN\NONAFIB TRAIN\chapman_non_afib"
SAVE_MODEL_PATH = "xgb_ecg_model.joblib"
MAX_SAMPLES_PER_CLASS = 500   # limit to 500 AFIB + 500 NONAFIB

# -----------------------------
# Helper Functions
# -----------------------------
def read_mat_ecg(filepath):
    """
    Reads .mat ECG file and returns 1D signal (first lead).
    """
    try:
        mat = scipy.io.loadmat(filepath)
        if "val" in mat:
            sig = np.squeeze(mat["val"])
        else:
            # Fallback: first numeric array
            for k, v in mat.items():
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    sig = np.squeeze(v)
                    break
            else:
                return None

        # If 2D (multi-lead), take first lead
        if sig.ndim == 2:
            sig = sig[0, :]
        return sig.astype(float)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def extract_features(ecg_signal, fs=500):
    """
    Extracts a compact set of features from ECG signal.
    """
    if ecg_signal is None or len(ecg_signal) < 100:
        return None

    # Normalize
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)

    # Statistical
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    max_val = np.max(ecg_signal)
    min_val = np.min(ecg_signal)
    skew_val = pd.Series(ecg_signal).skew()
    kurt_val = pd.Series(ecg_signal).kurt()

    # Frequency domain
    f, psd = welch(ecg_signal, fs=fs, nperseg=min(len(ecg_signal), 1024))
    power_low = np.trapz(psd[(f >= 0.5) & (f <= 5)])
    power_high = np.trapz(psd[(f >= 5) & (f <= 15)])
    ratio_power = power_high / (power_low + 1e-6)

    # RR interval features
    peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
    else:
        mean_rr, std_rr = 0, 0

    return np.array([
        mean_val, std_val, max_val, min_val, skew_val, kurt_val,
        power_low, power_high, ratio_power, mean_rr, std_rr
    ])


# -----------------------------
# Dataset Builder
# -----------------------------
def build_dataset():
    features, labels = [], []

    # Collect AFIB and NONAFIB file paths
    afib_files, nonafib_files = [], []
    for root, _, files in os.walk(AFIB_DIR):
        for f in files:
            if f.endswith(".mat"):
                afib_files.append(os.path.join(root, f))

    for root, _, files in os.walk(NONAFIB_DIR):
        for f in files:
            if f.endswith(".mat"):
                nonafib_files.append(os.path.join(root, f))

    # Limit samples per class
    afib_files = random.sample(afib_files, min(MAX_SAMPLES_PER_CLASS, len(afib_files)))
    nonafib_files = random.sample(nonafib_files, min(MAX_SAMPLES_PER_CLASS, len(nonafib_files)))

    print(f"📊 Using {len(afib_files)} AFIB and {len(nonafib_files)} NON-AFIB signals.")

    # Process AFIB
    for fpath in afib_files:
        sig = read_mat_ecg(fpath)
        feats = extract_features(sig)
        if feats is not None:
            features.append(feats)
            labels.append(1)

    # Process NON-AFIB
    for fpath in nonafib_files:
        sig = read_mat_ecg(fpath)
        feats = extract_features(sig)
        if feats is not None:
            features.append(feats)
            labels.append(0)

    X = np.array(features)
    y = np.array(labels)
    print(f"✅ Dataset ready: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


# -----------------------------
# Training Pipeline
# -----------------------------
def train_model():
    X, y = build_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optimized XGBoost parameters for AFib detection
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.2,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss"
    )

    print("\n🚀 Training XGBoost model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=["Non-AFib", "AFib"]))

    # Save the model and scaler
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': [
            'mean_val', 'std_val', 'max_val', 'min_val', 'skew_val', 'kurt_val',
            'power_low', 'power_high', 'ratio_power', 'mean_rr', 'std_rr'
        ]
    }, SAVE_MODEL_PATH)

    print(f"\n✅ Model saved to {SAVE_MODEL_PATH}")

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    train_model()
