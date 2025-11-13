import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt


# =====================================================
#                SIGNAL / FEATURE HELPERS
# =====================================================

def bandpass_filter(signal, fs, low=2.0, high=30.0, order=2):
    """
    Simple Butterworth band-pass filter.
    Keeps frequencies between low and high (Hz).
    fs = sampling rate (frames per second).
    """
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq

    if high_n >= 1.0:
        high_n = 0.99

    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def fft_peak_features(signal, fs, fmin=2.0, fmax=30.0):
    """
    Find dominant frequency and amplitude of the filtered signal
    inside [fmin, fmax] using the FFT.
    Returns:
      peak_freq, peak_amp, filtered_signal
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)

    filtered = bandpass_filter(signal, fs, low=fmin, high=fmax)

    spec = rfft(filtered)
    freqs = rfftfreq(len(filtered), d=1.0 / fs)
    mag = np.abs(spec)

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, np.nan, filtered

    idx = np.argmax(mag[mask])
    peak_freq = freqs[mask][idx]
    peak_amp = mag[mask][idx]

    return float(peak_freq), float(peak_amp), filtered


def cbp_sine_score(filtered_signal, fs, main_freq):
    """
    Very simple CBP-style feature:
    how similar the filtered motion is to a clean sine wave
    at the main beat frequency.

    Returns a value roughly between -1 and 1:
      - closer to 1: smooth, regular oscillation
      - closer to 0: noisy or irregular
    """
    if filtered_signal is None:
        return np.nan
    if np.isnan(main_freq) or main_freq <= 0:
        return np.nan

    sig = np.asarray(filtered_signal, dtype=float)
    if sig.size == 0:
        return np.nan

    t = np.arange(len(sig)) / float(fs)
    ideal = np.sin(2.0 * np.pi * main_freq * t)

    sig = sig - np.mean(sig)
    ideal = ideal - np.mean(ideal)

    sig_std = np.std(sig)
    ideal_std = np.std(ideal)
    if sig_std == 0 or ideal_std == 0:
        return np.nan

    sig = sig / sig_std
    ideal = ideal / ideal_std

    score = float(np.mean(sig * ideal))
    return score


def compute_global_features(frames, fps):
    """
    Compute global motion features from a video.

    Uses the mean brightness of each frame over time as
    a 1D signal, then extracts:
      - CBF (Hz)
      - FFT peak amplitude
      - variance
      - zero-crossing rate
      - simple CBP measures (sine score + amplitude)
    """
    # mean brightness per frame: shape (T,)
    mean_signal = frames.mean(axis=(1, 2))

    cbf, amp, filtered = fft_peak_features(mean_signal, fps, fmin=2.0, fmax=30.0)

    try:
        var_value = float(np.var(filtered))
        sign_changes = np.diff(np.sign(filtered)) != 0
        zcr_value = float(np.mean(sign_changes))
        cbp_score = cbp_sine_score(filtered, fps, cbf)
        cbp_amp = float(np.std(filtered))
    except Exception:
        var_value = np.nan
        zcr_value = np.nan
        cbp_score = np.nan
        cbp_amp = np.nan

    return {
        "global_cbf_hz": cbf,
        "global_peak_amp": amp,
        "global_var": var_value,
        "global_zcr": zcr_value,
        "global_cbp_sine_score": cbp_score,
        "global_cbp_amp": cbp_amp,
        # keep signals for plotting
        "raw_signal": mean_signal,
        "filtered_signal": filtered,
    }


def load_frames_from_file(path, max_frames=300):
    """
    Read up to max_frames grayscale frames from a video.
    Returns frames with shape (T, H, W) and fps.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        count += 1

        if max_frames is not None and count >= max_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video.")

    frames = np.stack(frames, axis=0)
    return frames, fps


def extract_features_from_uploaded_video(uploaded_file, max_frames=300):
    """
    Save the uploaded video to a temp file so OpenCV can read it,
    then extract global features.
    """
    suffix = ".avi"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()

    frames, fps = load_frames_from_file(tmp.name, max_frames=max_frames)
    feats = compute_global_features(frames, fps)
    return feats, fps, frames.shape[0]


# =====================================================
#             CBP / CBF SIMULATION HELPERS
# =====================================================

def simulate_cilia_signal(cbf_hz, duration_sec, fps, regularity, amplitude):
    """
    Create a fake cilia motion signal for the simulator.

    - cbf_hz: beats per second (Hz)
    - duration_sec: simulation duration in seconds
    - fps: samples per second
    - regularity: 0..1 (1 = very smooth, 0 = noisy/irregular)
    - amplitude: overall strength of motion
    """
    n_samples = int(duration_sec * fps)
    if n_samples < 10:
        n_samples = 10

    t = np.arange(n_samples) / float(fps)

    # base clean sine at chosen CBF
    base = np.sin(2.0 * np.pi * cbf_hz * t)

    # slow envelope to simulate small changes in strength
    envelope = 1.0 + (1.0 - regularity) * 0.5 * np.sin(2.0 * np.pi * 0.2 * t)

    # noise level grows as regularity drops
    noise_scale = (1.0 - regularity) * 0.8
    noise = np.random.normal(scale=noise_scale, size=n_samples)

    signal = amplitude * base * envelope + noise
    return t, signal


def draw_cilium_frame(angle_value, max_angle_deg=40):
    """
    Draw a simple stick cilium that tilts based on angle_value in [-1, 1].
    angle_value usually comes from a normalized signal sample.
    """
    angle_value = float(angle_value)
    angle_value = max(-1.0, min(1.0, angle_value))

    angle_deg = angle_value * max_angle_deg
    angle_rad = np.deg2rad(angle_deg)

    length = 1.0
    x0, y0 = 0.0, 0.0
    x1 = length * np.sin(angle_rad)
    y1 = length * np.cos(angle_rad)

    fig, ax = plt.subplots()
    ax.plot([x0, x1], [y0, y1], linewidth=5)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Cilium angle: {angle_deg:.1f}°")

    return fig


# =====================================================
#                  MODEL TRAINING
# =====================================================

@st.cache_data
def load_training_data(csv_path="labeled_cilia_video_dataset.csv"):
    """
    Load the labeled per-video dataset.
    If it has no 'class' column, derive it from 'label' using
    a simple Healthy vs PCD rule.
    """
    df = pd.read_csv(csv_path)

    if "class" not in df.columns:
        if "label" in df.columns:
            df["class"] = df["label"].apply(
                lambda x: "Healthy" if "Healthy" in str(x) else "PCD"
            )
        else:
            raise RuntimeError("Dataset needs a 'class' or 'label' column.")
    return df


@st.cache_resource
def train_random_forest_model(csv_path="labeled_cilia_video_dataset.csv"):
    """
    Train a RandomForest classifier on global features.
    Uses both CBF metrics and simple CBP-related features.
    """
    df = load_training_data(csv_path)

    feature_cols = [
        "global_cbf_hz",
        "global_peak_amp",
        "global_var",
        "global_zcr",
        "global_cbp_sine_score",
        "global_cbp_amp",
    ]

    df_clean = df.dropna(subset=feature_cols + ["class"])
    if len(df_clean) < 3:
        raise RuntimeError("Not enough training samples with complete features.")

    X = df_clean[feature_cols]
    y = df_clean["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)

    return model, feature_cols, acc, report_text


# =====================================================
#                      STREAMLIT APP
# =====================================================

def main():
    st.set_page_config(page_title="Cilia Motion Classifier Demo", layout="centered")

    # ---------------- Sidebar ----------------
    st.sidebar.title("About this app")
    st.sidebar.write(
        """
        This app is a **technical demo** for analysing cilia motion:

        1. Extracts motion features (CBF + simple CBP-like measures)
        2. Trains a small Random Forest classifier on a labeled dataset
        3. Predicts whether a new video looks more **Healthy** or **PCD-like**

        It is designed for learning and prototyping, **not** for real diagnosis.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Backend: Python, OpenCV, SciPy, scikit-learn, Streamlit.")

    # ---------------- Header ----------------
    st.title("🧬 Cilia Motion Classifier & Simulator")

    st.write(
        "This demo takes short high-speed cilia videos, extracts signal features "
        "such as ciliary beat frequency (CBF) and simple ciliary beating pattern (CBP) "
        "measures, then uses a Random Forest model to separate **Healthy** from "
        "**PCD-like** motion in a small dataset."
    )

    # ---------------- Train / load model ----------------
    try:
        model, feature_cols, acc, report_txt = train_random_forest_model(
            "labeled_cilia_video_dataset.csv"
        )
        st.success(f"Random Forest trained. Test accuracy: **{acc*100:.1f}%**")
    except Exception as e:
        st.error(f"Could not train model: {e}")
        return

    with st.expander("Show training classification report"):
        st.text(report_txt)

    # ---------------- Upload section ----------------
    st.markdown("---")
    st.subheader("📤 Upload a cilia video for analysis")

    uploaded_video = st.file_uploader(
        "Upload a video file (.avi, .mp4, .mov)", type=["avi", "mp4", "mov"]
    )

    max_frames = st.slider(
        "Max frames to analyse from the video",
        min_value=100,
        max_value=400,
        value=256,
        step=16,
    )

    if uploaded_video is not None:
        st.info("Processing uploaded video...")
        try:
            feats, fps, n_frames = extract_features_from_uploaded_video(
                uploaded_video, max_frames=max_frames
            )
        except Exception as e:
            st.error(f"Error reading or processing video: {e}")
            return

        st.write(f"**Video info:** {n_frames} frames used at {fps:.1f} fps")

        # Strip out raw arrays for the JSON view
        feats_for_display = {
            k: v for k, v in feats.items()
            if k not in ["raw_signal", "filtered_signal"]
        }

        st.write("### Extracted global features")
        st.json(feats_for_display)

        # Plot raw vs filtered signal if available
        raw = feats.get("raw_signal", None)
        flt = feats.get("filtered_signal", None)

        if raw is not None and flt is not None:
            st.write("### Motion signal over time (global mean intensity)")
            t = np.arange(len(raw)) / float(fps)

            fig_sig, ax_sig = plt.subplots()
            ax_sig.plot(t, raw, label="Raw mean intensity")
            ax_sig.plot(t, flt, label="Filtered (2–30 Hz)")
            ax_sig.set_xlabel("Time (seconds)")
            ax_sig.set_ylabel("Signal")
            ax_sig.grid(True)
            ax_sig.legend()
            st.pyplot(fig_sig)

        # Prepare data for prediction
        try:
            X_new = pd.DataFrame([feats_for_display])[feature_cols]
        except KeyError as e:
            st.error(f"Missing required feature for prediction: {e}")
            return

        pred_class = model.predict(X_new)[0]
        pred_prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            class_to_idx = {c: i for i, c in enumerate(model.classes_)}
            pred_prob = proba[class_to_idx[pred_class]]

        st.subheader("🩺 Model prediction")

        if pred_class == "Healthy":
            msg = "Prediction: **Healthy-like cilia motion**"
            if pred_prob is not None:
                msg += f" (confidence ~{pred_prob*100:.1f}%)"
            st.success(msg)
        else:
            msg = "Prediction: **PCD-like cilia motion**"
            if pred_prob is not None:
                msg += f" (confidence ~{pred_prob*100:.1f}%)"
            st.error(msg)

        # Quick verbal interpretation of CBF and CBP
        cbf_val = feats_for_display.get("global_cbf_hz", np.nan)
        cbp_score = feats_for_display.get("global_cbp_sine_score", np.nan)
        cbp_amp = feats_for_display.get("global_cbp_amp", np.nan)

        st.markdown("### Quick interpretation")
        bullets = []

        if not np.isnan(cbf_val):
            bullets.append(f"- Estimated CBF ≈ **{cbf_val:.1f} Hz**.")
        if not np.isnan(cbp_score):
            bullets.append(
                f"- CBP sine score ≈ **{cbp_score:.2f}** "
                "(closer to 1 means very regular, sine-like beating)."
            )
        if not np.isnan(cbp_amp):
            bullets.append(
                f"- CBP amplitude ≈ **{cbp_amp:.3f}** "
                "(larger values indicate stronger intensity oscillations)."
            )

        if bullets:
            st.write("\n".join(bullets))
        else:
            st.write("Could not compute CBF / CBP features for this video.")

        st.caption(
            "Reminder: this is a small proof-of-concept model trained on a limited dataset. "
            "It is for educational and technical demonstration only."
        )

    else:
        st.info("Upload a video file above to see features and a prediction.")

    # =================================================
    #             INTERACTIVE SIMULATION LAB
    # =================================================

    st.markdown("---")
    st.header("🎮 Cilia Motion Simulation Lab")

    st.write(
        "This section simulates a 1D cilia motion signal using simple sliders. "
        "It is not based on real data, but it helps build intuition about how "
        "CBF and CBP-like changes affect the signal shape."
    )

    col1, col2 = st.columns(2)

    with col1:
        sim_cbf = st.slider(
            "Simulated CBF (beats per second)",
            min_value=2.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
        )
        sim_reg = st.slider(
            "Regularity (0 = very irregular, 1 = very smooth)",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
        )
        sim_amp = st.slider(
            "Motion amplitude",
            min_value=0.2,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )
        sim_duration = st.slider(
            "Simulation duration (seconds)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.5,
        )

    with col2:
        st.write("Presets (quick examples):")
        preset = st.radio(
            "Choose a preset",
            options=["None", "Healthy-like", "PCD-like"],
            index=0,
        )

        if preset == "Healthy-like":
            sim_cbf = 10.0
            sim_reg = 0.9
            sim_amp = 1.2
        elif preset == "PCD-like":
            sim_cbf = 4.0
            sim_reg = 0.3
            sim_amp = 0.8

        st.write(
            "- **Healthy-like**: higher CBF, smooth beating, strong amplitude\n"
            "- **PCD-like**: lower CBF, irregular pattern, weaker amplitude"
        )

    # Simulate signal
    sim_fps = 100.0
    t_sim, sig_sim = simulate_cilia_signal(
        cbf_hz=sim_cbf,
        duration_sec=sim_duration,
        fps=sim_fps,
        regularity=sim_reg,
        amplitude=sim_amp,
    )

    st.subheader("Simulated brightness / motion signal")
    fig_sim, ax_sim = plt.subplots()
    ax_sim.plot(t_sim, sig_sim)
    ax_sim.set_xlabel("Time (seconds)")
    ax_sim.set_ylabel("Simulated signal")
    ax_sim.grid(True)
    st.pyplot(fig_sim)

    st.subheader("Toy cilium view (single frame)")

    idx = st.slider(
        "Pick a time index inside the simulated signal",
        min_value=0,
        max_value=len(sig_sim) - 1,
        value=len(sig_sim) // 2,
    )

    frame_val = sig_sim[idx]
    sig_std = np.std(sig_sim) if np.std(sig_sim) > 0 else 1.0
    norm_val = frame_val / (3.0 * sig_std)  # compress to a reasonable range

    fig_cil = draw_cilium_frame(norm_val)
    st.pyplot(fig_cil)

    st.caption(
        "In this simple visualization, the cilium is drawn as a line that tilts "
        "according to the simulated signal at one instant in time."
    )


if __name__ == "__main__":
    main()
