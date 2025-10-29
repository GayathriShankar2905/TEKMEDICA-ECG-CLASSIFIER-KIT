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
import os
import joblib

# Dynamically locate model file
MODEL_PATH = os.path.join(os.getcwd(), "xgb_ecg_model.joblib")

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}: {e}")
    st.stop()

PAGE_TITLE = "TekMedica — ECG Classification Kit"
PAGE_ICON = "🫀"

# -------- PAGE SETUP --------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# -------- LOAD MODEL --------
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler", None)
    feature_names = model_bundle.get("feature_names", None)
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}: {e}")
    st.stop()

# -------- STYLES (Light Mode, modern) --------
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

    html, body, [class*="css"]  {
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

# -------- LAYOUT: HERO + SIDEBAR --------
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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Upload & Predict", "Model Info", "About TekMedica"),
)

# -------- HELPER: READ .mat robustly (handles 'val' with multi-lead) --------
def read_mat_signal_from_uploader(uploaded_file):
    """
    Save uploaded file to temp, load with scipy.io.loadmat,
    search for 'val' or the first numeric array. Return 1D numpy array.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        mat = scipy.io.loadmat(tmp_path)
        os.remove(tmp_path)
    except Exception as e:
        st.error(f"Failed to read .mat: {e}")
        return None

    # Prefer 'val' key (common in ECG collections)
    if "val" in mat:
        arr = np.squeeze(mat["val"])
        # if multi-lead shaped (leads x samples), pick lead 0 by default
        if arr.ndim == 2:
            return arr  # return full 2D so UI can pick lead
        elif arr.ndim == 1:
            return arr
    # fallback: pick first ndarray with numeric dtype and length > 50
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            arr = np.squeeze(v)
            if arr.ndim in (1, 2) and arr.size > 50:
                return arr
    return None

# -------- FEATURE EXTRACTION (same features used for training) --------
def extract_features(ecg_signal, fs=360):
    if ecg_signal is None:
        return None
    ecg_signal = np.asarray(ecg_signal).astype(float)
    if ecg_signal.size < 100:
        return None

    # Normalize
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)

    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    max_val = np.max(ecg_signal)
    min_val = np.min(ecg_signal)
    skew_val = pd.Series(ecg_signal).skew()
    kurt_val = pd.Series(ecg_signal).kurt()

    f, psd = welch(ecg_signal, fs=fs, nperseg=min(len(ecg_signal), 1024))
    power_low = np.trapz(psd[(f >= 0.5) & (f <= 5)]) if np.any((f >= 0.5) & (f <= 5)) else 0.0
    power_high = np.trapz(psd[(f >= 5) & (f <= 15)]) if np.any((f >= 5) & (f <= 15)) else 0.0
    ratio_power = power_high / (power_low + 1e-8)

    peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.6))
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
    else:
        mean_rr = 0.0
        std_rr = 0.0

    feats = np.array([
        mean_val, std_val, max_val, min_val, skew_val, kurt_val,
        power_low, power_high, ratio_power, mean_rr, std_rr
    ], dtype=float)
    return feats

# -------- PAGES --------
if page == "Upload & Predict":
    # Upload card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload ECG (.mat only)")
    st.write("Upload a `.mat` file. This app will extract features and use the backend XGBoost model to classify AFib vs Non-AFib.")
    uploaded_file = st.file_uploader("Choose a .mat ECG file", type=["mat"])

    # sampling rate input (some .mat do not include fs info)
    fs_input = st.number_input("Sampling frequency (Hz)", min_value=50, max_value=2000, value=360, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        arr = read_mat_signal_from_uploader(uploaded_file)
        if arr is None:
            st.error("No valid numeric array found inside the .mat file.")
        else:
            # If arr is 2D (leads x samples), let user choose lead
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                n_leads = arr.shape[0]
                lead_idx = st.slider("Select lead (row) to analyze", min_value=1, max_value=n_leads, value=1)
                ecg_1d = arr[lead_idx - 1, :].astype(float)
            else:
                ecg_1d = np.squeeze(arr).astype(float)
                lead_idx = None

            # Show summary info
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Signal summary")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.write("**Length**")
                st.write(f"{ecg_1d.size:,} samples")
            with c2:
                st.write("**Sampling rate**")
                st.write(f"{fs_input} Hz")
            with c3:
                duration = ecg_1d.size / fs_input
                st.write("**Duration**")
                st.write(f"{duration:.1f} s")
            st.markdown('</div>', unsafe_allow_html=True)

            # Plot ECG (first N samples) and zoom controls
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("ECG Preview")
            preview_length = st.slider("Preview duration (seconds)", min_value=2, max_value=min(30, int(ecg_1d.size//fs_input)), value=6)
            n_samples_preview = int(preview_length * fs_input)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(ecg_1d[:n_samples_preview], color="#1f77b4", linewidth=1)
            ax.set_title("ECG (preview)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            sns.despine(trim=True, left=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # Extract features and predict
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Extract & Predict")
            with st.spinner("Extracting features and running model..."):
                features = extract_features(ecg_1d, fs=fs_input)
                if features is None:
                    st.error("Feature extraction failed (signal too short or invalid).")
                else:
                    # display feature vector
                    if feature_names is not None and len(feature_names) == len(features):
                        feat_df = pd.DataFrame([features], columns=feature_names)
                    else:
                        # generic names
                        feat_cols = [f"f{i+1}" for i in range(len(features))]
                        feat_df = pd.DataFrame([features], columns=feat_cols)

                    st.write("**Feature vector (values):**")
                    st.dataframe(feat_df.T, height=260)

                    # scale if scaler available
                    X_input = features.reshape(1, -1)
                    if scaler is not None:
                        try:
                            X_input = scaler.transform(X_input)
                        except Exception:
                            # fallback: use raw features
                            pass

                    y_pred = model.predict(X_input)[0]
                    prob_arr = model.predict_proba(X_input)[0]
                    # probability for predicted class
                    pred_prob = prob_arr[y_pred]
                    label = "Atrial Fibrillation (AFib)" if int(y_pred) == 1 else "Normal (Non-AFib)"
                    color = "#d62828" if int(y_pred) == 1 else "#198754"

                    st.markdown(f"<div style='text-align:center; margin:8px 0;'>"
                                f"<span class='result-badge' style='background:{color};'>{label}</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center; color:#334155; margin-bottom:6px;'>Confidence: <strong>{pred_prob*100:.2f}%</strong></div>", unsafe_allow_html=True)

                    # show class probabilities
                    probs_df = pd.DataFrame({
                        "Class": ["Non-AFib (0)", "AFib (1)"],
                        "Probability": [prob_arr[0], prob_arr[1]]
                    })
                    st.write("Prediction probabilities:")
                    st.bar_chart(probs_df.set_index("Class"))

                    # option to save results as CSV
                    if st.button("Download feature vector as CSV"):
                        tmp_csv = feat_df.to_csv(index=False)
                        st.download_button("Download CSV", tmp_csv, file_name="ecg_features.csv", mime="text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model Info":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Information & Feature Importances")
    # model summary
    st.write("**Model type:**", type(model).__name__)
    if hasattr(model, "get_params"):
        params = model.get_params()
        # show a subset
        display_params = {k: params[k] for k in list(params)[:12]}
        st.json(display_params)

    # feature importance if available
    try:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            # fallback to feature score
            fscore = booster.get_score(importance_type="weight")
            # map to list
            if fscore:
                # create importances aligned with feature_names
                importances = np.zeros(len(feature_names))
                for k, v in fscore.items():
                    # k like 'f0','f1' etc.
                    idx = int(k.strip("f"))
                    importances[idx] = v
        if importances is not None and feature_names is not None:
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False)
            st.write("Top features used by the model:")
            st.dataframe(fi_df.head(10).reset_index(drop=True))
            # simple bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(fi_df["feature"].head(10)[::-1], fi_df["importance"].head(10)[::-1], color="#0b66c2")
            ax.set_xlabel("Importance")
            ax.set_title("Feature importances (top 10)")
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "About TekMedica":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About TekMedica & This Demo")
    st.markdown("""
    **TekMedica** — Student club at SRM Institute of Science and Technology.
    This educational demo demonstrates a simple ML pipeline for ECG classification:
    - Feature extraction from ECG waveforms  
    - Classification using a pre-trained XGBoost model (AFib vs Non-AFib)  
    - Explanation and visualization for learning purposes

    **Built by:** Gayathri S.H @TEKMEDICA  
    **Contact:** pcmjs.gayathri@gmail.com
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER CREDIT --------
st.markdown(
    """
    <div style="margin-top:18px; text-align:center; color:#475569;">
        <hr style="border:1px solid #e6eef8;">
        <div class="credit">Built by <strong>Gayathri S.H @TEKMEDICA</strong> — School of Bioengineering, SRM Institute of Science and Technology</div>
    </div>
    """,
    unsafe_allow_html=True,
)
