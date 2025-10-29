# ============================================================
# app.py — TekMedica ECG Classification Kit (Light Mode, Polished)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import find_peaks, welch
import joblib
import matplotlib.pyplot as plt
import tempfile
import os
import seaborn as sns

# -------- CONFIG --------
PAGE_TITLE = "TekMedica — ECG Classification Kit"
PAGE_ICON = "🫀"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# -------- LOAD MODEL (fixed single section) --------
MODEL_PATH = os.path.join(os.getcwd(), "xgb_ecg_model.joblib")

try:
    model_bundle = joblib.load(MODEL_PATH)

    if isinstance(model_bundle, dict):
        model = model_bundle.get("model", None)
        scaler = model_bundle.get("scaler", None)
        feature_names = model_bundle.get("feature_names", None)
    else:
        model = model_bundle
        scaler = None
        feature_names = None

    if model is None:
        raise ValueError("No valid model found inside joblib file.")

    st.success(f"✅ Model loaded successfully from: {MODEL_PATH}")

except Exception as e:
    st.error(f"❌ Error loading model from {MODEL_PATH}: {e}")
    st.stop()

# -------- STYLES --------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    :root{
      --tek-blue: #002b5b;
      --tek-cyan: #06b6d4;
      --card-bg: rgba(255,255,255,0.9);
      --muted: #6b7280;
    }
    html, body, [class*="css"] {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(180deg,#f5fbff 0%, #ffffff 60%);
    }
    .hero {
      background: linear-gradient(90deg, rgba(0,43,91,0.95), rgba(6,182,212,0.85));
      color: white;
      padding: 28px;
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.12);
      margin-bottom: 18px;
    }
    .hero h1 { margin: 0; font-weight:700; letter-spacing:0.2px; }
    .hero h4 { margin: 6px 0 0 0; font-weight:500; color: rgba(255,255,255,0.95); }
    .card {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 18px rgba(10,20,40,0.06);
      margin-bottom: 18px;
    }
    .small-muted { color: var(--muted); font-size:0.9rem; }
    .credit { color: #475569; font-size:0.85rem; text-align:center; margin-top:12px; }
    .result-badge {
      display:inline-block;
      padding:12px 20px;
      border-radius:999px;
      font-weight:700;
      color:white;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- HEADER --------
st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h1>🫀 TEKMEDICA — ECG Classification Kit</h1>
          <h4>School of Bioengineering, SRM Institute of Science and Technology</h4>
        </div>
        <div style="text-align:right">
          <div style="font-size:0.95rem; color:rgba(255,255,255,0.9)">Upload → Analyze → Visualize → Classify</div>
          <div style="margin-top:10px; font-size:0.9rem; color:rgba(255,255,255,0.9)">Built by <strong>Gayathri S.H @TEKMEDICA</strong></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------- SIDEBAR --------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Upload & Predict", "Model Info", "About TekMedica"))

# -------- UTIL: READ .mat ECG --------
def read_mat_signal_from_uploader(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        mat = scipy.io.loadmat(tmp_path)
        os.remove(tmp_path)
    except Exception as e:
        st.error(f"Failed to read .mat: {e}")
        return None

    if "val" in mat:
        arr = np.squeeze(mat["val"])
        if arr.ndim == 2:
            return arr
        elif arr.ndim == 1:
            return arr
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            arr = np.squeeze(v)
            if arr.ndim in (1, 2) and arr.size > 50:
                return arr
    return None

# -------- FEATURE EXTRACTION --------
def extract_features(ecg_signal, fs=360):
    if ecg_signal is None or ecg_signal.size < 100:
        return None
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    mean_val, std_val = np.mean(ecg_signal), np.std(ecg_signal)
    max_val, min_val = np.max(ecg_signal), np.min(ecg_signal)
    skew_val, kurt_val = pd.Series(ecg_signal).skew(), pd.Series(ecg_signal).kurt()
    f, psd = welch(ecg_signal, fs=fs, nperseg=min(len(ecg_signal), 1024))
    power_low = np.trapz(psd[(f >= 0.5) & (f <= 5)]) if np.any((f >= 0.5) & (f <= 5)) else 0.0
    power_high = np.trapz(psd[(f >= 5) & (f <= 15)]) if np.any((f >= 5) & (f <= 15)) else 0.0
    ratio_power = power_high / (power_low + 1e-8)
    peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.6))
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr, std_rr = np.mean(rr_intervals), np.std(rr_intervals)
    else:
        mean_rr, std_rr = 0.0, 0.0
    return np.array([
        mean_val, std_val, max_val, min_val, skew_val, kurt_val,
        power_low, power_high, ratio_power, mean_rr, std_rr
    ])

# -------- MAIN PAGE: UPLOAD & PREDICT --------
if page == "Upload & Predict":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload ECG (.mat only)")
    uploaded_file = st.file_uploader("Choose a .mat ECG file", type=["mat"])
    fs_input = st.number_input("Sampling frequency (Hz)", min_value=50, max_value=2000, value=360, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        arr = read_mat_signal_from_uploader(uploaded_file)
        if arr is None:
            st.error("No valid numeric array found inside the .mat file.")
        else:
            if arr.ndim == 2:
                lead_idx = st.slider("Select lead", 1, arr.shape[0], 1)
                ecg_1d = arr[lead_idx - 1, :]
            else:
                ecg_1d = arr

            # Summary
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Signal summary")
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Length:** {ecg_1d.size:,} samples")
            c2.write(f"**Sampling rate:** {fs_input} Hz")
            c3.write(f"**Duration:** {ecg_1d.size / fs_input:.1f} s")
            st.markdown('</div>', unsafe_allow_html=True)

            # Plot
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("ECG Preview")
            preview_length = st.slider("Preview duration (s)", 2, min(30, int(ecg_1d.size // fs_input)), 6)
            n_samples = int(preview_length * fs_input)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(ecg_1d[:n_samples], color="#1f77b4", linewidth=1)
            ax.set_title("ECG (preview)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            sns.despine(trim=True, left=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # Predict
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Extract & Predict")
            with st.spinner("Running model..."):
                features = extract_features(ecg_1d, fs_input)
                if features is None:
                    st.error("Feature extraction failed.")
                else:
                    X_input = features.reshape(1, -1)
                    if scaler is not None:
                        try:
                            X_input = scaler.transform(X_input)
                        except Exception:
                            pass

                    y_pred = model.predict(X_input)[0]
                    prob_arr = model.predict_proba(X_input)[0]
                    pred_prob = prob_arr[y_pred]
                    label = "Atrial Fibrillation (AFib)" if y_pred == 1 else "Normal (Non-AFib)"
                    color = "#d62828" if y_pred == 1 else "#198754"

                    st.markdown(
                        f"<div style='text-align:center;margin:8px 0;'>"
                        f"<span class='result-badge' style='background:{color};'>{label}</span></div>"
                        f"<div style='text-align:center;color:#334155;'>Confidence: <strong>{pred_prob*100:.2f}%</strong></div>",
                        unsafe_allow_html=True
                    )

                    probs_df = pd.DataFrame({"Class": ["Non-AFib (0)", "AFib (1)"], "Probability": prob_arr})
                    st.bar_chart(probs_df.set_index("Class"))

            st.markdown('</div>', unsafe_allow_html=True)

# -------- PAGE: MODEL INFO --------
elif page == "Model Info":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Information & Feature Importances")
    st.write("**Model type:**", type(model).__name__)
    if hasattr(model, "get_params"):
        st.json({k: v for k, v in list(model.get_params().items())[:12]})
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(fi_df["feature"].head(10)[::-1], fi_df["importance"].head(10)[::-1], color="#0b66c2")
            ax.set_xlabel("Importance")
            ax.set_title("Top 10 Features")
            st.pyplot(fig)
        else:
            st.info("Feature importances unavailable.")
    except Exception as e:
        st.warning(f"Could not compute importances: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------- PAGE: ABOUT --------
elif page == "About TekMedica":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About TekMedica & This Demo")
    st.markdown("""
    **TekMedica** — Student club at SRM Institute of Science and Technology.  
    This educational app demonstrates:
    - ECG waveform feature extraction  
    - XGBoost-based classification (AFib vs Non-AFib)  
    - Visual analytics and explainability  

    **Built by:** Gayathri S.H @TEKMEDICA  
    **Contact:** pcmjs.gayathri@gmail.com
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown(
    """
    <div style="margin-top:18px; text-align:center; color:#475569;">
        <hr style="border:1px solid #e6eef8;">
        <div class="credit">
            Built by <strong>Gayathri S.H @TEKMEDICA</strong> — School of Bioengineering, SRM Institute of Science and Technology
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
