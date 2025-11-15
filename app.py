import os
import tempfile
from datetime import datetime   # NEW: for logging questions

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from openai import OpenAI


# ==========================
#   FEATURE EXPLANATIONS
# ==========================

FEATURE_HELP = {
    "mode": (
        "Modes change how information is shown:\n"
        "- Clinician: compact view with key numbers (CBF, regularity, amplitude).\n"
        "- Student: more text and step-by-step explanations.\n"
        "- Researcher: shows raw feature table and the full classification report.\n"
        "- Tour: guided explanation of what each part of the app does."
    ),
    "cilia_bot": (
        "CiliaBot is a small chatbot that explains PCD, motile cilia, CBF, CBP and what "
        "the graphs mean in simple language. It is for education only and does NOT give "
        "medical advice or diagnosis."
    ),
    "cbf": (
        "CBF (Ciliary Beat Frequency) is how many times per second the cilia beat. "
        "For example, 10 Hz means about 10 beats each second. In this app, CBF is "
        "estimated from brightness changes in the video using a band-pass filter and the FFT."
    ),
    "cbp_score": (
        "The CBP (ciliary beating pattern) sine score is a rough measure of how regular "
        "the motion is. A value near 1 means the motion looks like a clean, smooth sine "
        "wave. Values nearer 0 mean more irregular or noisy beating."
    ),
    "cbp_amp": (
        "The CBP amplitude is the size of the oscillation in the filtered signal. "
        "A larger amplitude means stronger brightness changes over time, which usually "
        "corresponds to stronger or larger-range beating."
    ),
    "zcr": (
        "ZCR (zero-crossing rate) counts how often the signal crosses zero. "
        "If the signal crosses zero very often, it may indicate faster or more jittery "
        "changes. Here it is just one simple feature among many."
    ),
    "var": (
        "Variance measures how spread out the filtered signal values are. Higher variance "
        "means the motion signal has larger changes in intensity over time."
    ),
    "prediction": (
        "The Random Forest classifier looks at several global features (CBF, variance, "
        "zero-crossing rate, regularity score, amplitude) and compares them to labelled "
        "training examples. It then assigns a label such as 'Healthy-like' or 'PCD-like'. "
        "This is a research / teaching demo only, not a diagnostic tool."
    ),
    "signals": (
        "The raw signal is the average brightness of each frame over time. This tells us "
        "how the overall intensity changes during the video. The filtered signal keeps "
        "only frequencies in a chosen band (for example 2–30 Hz) so we can focus on the "
        "cilia beating and remove slow drifts or very fast noise."
    ),
    "fft": (
        "The FFT (Fast Fourier Transform) converts the time-based signal into frequency "
        "space. Peaks in the FFT show which beat frequencies dominate the motion. The "
        "highest peak inside a chosen range (for example 2–30 Hz) is used as the estimated CBF."
    ),
    "sim_sliders": (
        "The simulation sliders let you play with a synthetic cilia signal:\n"
        "- CBF: how fast the cilia beat (Hz).\n"
        "- Regularity: 1 = smooth, periodic motion; 0 = very irregular.\n"
        "- Amplitude: overall strength of the oscillation.\n"
        "- Duration: how long the simulated signal lasts.\n"
        "This is not real patient data, just a teaching tool."
    ),
    "sim_cilium": (
        "The toy cilium view shows a single line that tilts according to the simulated "
        "signal at one moment in time. It gives an intuitive feel of how changes in the "
        "signal would look as a physical cilium moving."
    ),
    "multi_cilia": (
        "The multi-cilia visualizer draws many simple sine waves with different random "
        "phases. This is a cartoon-style sketch of a field of cilia beating together "
        "at a chosen frequency. It helps build intuition about synchronous vs slightly "
        "out-of-phase beating; it is not real data."
    ),
}


# ==========================
#   TOUR STEPS
# ==========================

TOUR_STEPS = [
    {
        "title": "Overview of the app",
        "body": (
            "This app has three main pieces:\n"
            "1. Analyse video: upload a cilia video and extract features like CBF and CBP.\n"
            "2. Simulation lab: play with a synthetic cilia signal to build intuition.\n"
            "3. Multi-cilia visualizer: see a cartoon of many cilia beating with a chosen CBF.\n\n"
            "Use this tour to read what each part does, then try it yourself."
        ),
    },
    {
        "title": "Uploading a video",
        "body": (
            "Go to the 'Analyse video' tab. Upload a short high-speed cilia video. "
            "The app converts the frames to grayscale, computes the mean brightness per frame, "
            "and treats that as a 1D signal. This is what all feature calculations are based on."
        ),
    },
    {
        "title": "Key global features",
        "body": (
            "After uploading, you see global features:\n"
            "- CBF (Hz): estimated dominant beat frequency.\n"
            "- Regularity score: how sine-like the motion looks.\n"
            "- Amplitude: how strong the oscillations are.\n"
            "- Variance and zero-crossing rate: additional signal statistics.\n\n"
            "In Student mode you see these in a table; in Clinician mode they are shown as metrics."
        ),
    },
    {
        "title": "Model prediction",
        "body": (
            "The Random Forest classifier takes the global features and compares them to a small "
            "labelled dataset. It outputs categories like 'Healthy-like' or 'PCD-like'. "
            "This is only an educational demo; it is not a clinical tool."
        ),
    },
    {
        "title": "Signals and FFT",
        "body": (
            "Below the prediction, you can see plots of:\n"
            "- Raw signal: mean brightness per frame.\n"
            "- Filtered signal: band-pass filtered between 2–30 Hz.\n"
            "You can also tick the FFT checkbox to see which frequencies dominate. "
            "The highest peak in the chosen band is used as the CBF estimate."
        ),
    },
    {
        "title": "Simulation lab",
        "body": (
            "In the 'Simulation lab' tab, you can simulate a 1D cilia motion signal by setting:\n"
            "- CBF, regularity, amplitude, and duration.\n"
            "The plot shows how the signal changes over time, and the toy cilium view turns that "
            "signal into a simple rotating line. This builds intuition for how signals map to motion."
        ),
    },
    {
        "title": "Multi-cilia visualizer",
        "body": (
            "In the 'Multi-cilia visualizer' tab, you can adjust the number of cilia and a common CBF. "
            "The plot shows many simple waveforms stacked vertically with random phase shifts, which "
            "is a cartoon of a field of beating cilia. It helps students imagine synchrony vs small phase shifts."
        ),
    },
]


# ==========================
#   SMALL HELPERS
# ==========================

def explain_button(key: str, label: str = "Explain this"):
    """Show an 'Explain this' button and an info box when clicked."""
    if st.button(label, key=f"explain_{key}"):
        st.info(FEATURE_HELP.get(key, "No explanation available for this item yet."))


def log_ciliabot_question(question: str, mode: str):
    """Log CiliaBot questions to a local CSV file so you can inspect them later."""
    question = question.strip()
    if not question:
        return

    log_path = "cilia_bot_log.csv"
    row = pd.DataFrame(
        [{
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "question": question,
        }]
    )
    if os.path.exists(log_path):
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, mode="w", header=True, index=False)


# ==========================
#   GPT Chatbot
# ==========================

def ask_gpt(question: str) -> str:
    """GPT-powered CiliaBot (no medical advice)."""
    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        return "⚠️ OpenAI API key is not configured. CiliaBot is disabled."

    try:
        client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are CiliaBot, a friendly explainer for clinicians and students. "
                        "Explain PCD, motile cilia, CBF, CBP, and the features of this app in "
                        "simple, accurate language. Always say this is for education only and "
                        "not medical advice or diagnosis."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=250,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error contacting OpenAI: {e}"


# ==========================
#   Signal processing
# ==========================

def bandpass_filter(signal, fs, low=2.0, high=30.0, order=2):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def fft_peak_features(signal, fs, fmin=2.0, fmax=30.0):
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
    peak_freq = float(freqs[mask][idx])
    peak_amp = float(mag[mask][idx])

    return peak_freq, peak_amp, filtered


def cbp_sine_score(filtered_signal, fs, main_freq):
    if filtered_signal is None or np.isnan(main_freq) or main_freq <= 0:
        return np.nan

    sig = np.asarray(filtered_signal, dtype=float)
    sig = sig - np.mean(sig)
    t = np.arange(len(sig)) / float(fs)

    ideal = np.sin(2.0 * np.pi * main_freq * t)
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
    mean_signal = frames.mean(axis=(1, 2))

    cbf, amp, filtered = fft_peak_features(mean_signal, fps, fmin=2.0, fmax=30.0)
    zcr = float(np.mean(np.diff(np.sign(filtered)) != 0))
    var = float(np.var(filtered))
    cbp_score = cbp_sine_score(filtered, fps, cbf)
    cbp_amp = float(np.std(filtered))

    return {
        "global_cbf_hz": cbf,
        "global_peak_amp": amp,
        "global_var": var,
        "global_zcr": zcr,
        "global_cbp_sine_score": cbp_score,
        "global_cbp_amp": cbp_amp,
        "raw_signal": mean_signal,
        "filtered_signal": filtered,
    }


# ==========================
#   Video helpers
# ==========================

def load_frames_from_file(path, max_frames=300):
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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    tmp.write(uploaded_file.read())
    tmp.close()

    frames, fps = load_frames_from_file(tmp.name, max_frames=max_frames)
    feats = compute_global_features(frames, fps)
    return feats, fps, frames.shape[0]


# ==========================
#   Simulation helpers
# ==========================

def simulate_cilia_signal(cbf_hz, duration_sec, fps, regularity, amplitude):
    n = int(duration_sec * fps)
    if n < 10:
        n = 10

    t = np.arange(n) / float(fps)
    base = np.sin(2.0 * np.pi * cbf_hz * t)
    noise = np.random.normal(scale=(1.0 - regularity) * 0.8, size=n)
    envelope = 1.0 + (1.0 - regularity) * 0.4 * np.sin(2.0 * np.pi * 0.3 * t)

    signal = amplitude * base * envelope + noise
    return t, signal


def draw_cilium_frame(value, max_angle_deg=40):
    v = float(np.clip(value, -1.0, 1.0))
    angle = v * max_angle_deg
    rad = np.deg2rad(angle)

    x1 = np.sin(rad)
    y1 = np.cos(rad)

    fig, ax = plt.subplots()
    ax.plot([0, x1], [0, y1], linewidth=6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Cilium angle: {angle:.1f}°")
    return fig


# ==========================
#   Model training
# ==========================

@st.cache_data
def load_training_data(csv_path="labeled_cilia_video_dataset.csv"):
    df = pd.read_csv(csv_path)
    if "class" not in df.columns:
        df["class"] = df["label"].apply(
            lambda x: "Healthy" if "Healthy" in str(x) else "PCD"
        )
    return df


@st.cache_resource
def train_model(csv_path="labeled_cilia_video_dataset.csv"):
    df = load_training_data(csv_path)

    feat_cols = [
        "global_cbf_hz",
        "global_peak_amp",
        "global_var",
        "global_zcr",
        "global_cbp_sine_score",
        "global_cbp_amp",
    ]

    df_clean = df.dropna(subset=feat_cols + ["class"])
    if len(df_clean) < 3:
        raise RuntimeError("Not enough training samples with complete features.")

    X = df_clean[feat_cols].values
    y = df_clean["class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)

    return model, feat_cols, acc, report_text


# ==========================
#   Main UI
# ==========================

def main():
    st.set_page_config(page_title="Cilia Motion Classifier Pro", layout="wide")
    st.title("🧬 Cilia Motion Classifier & Simulation Lab")

    # Sidebar: modes + chatbot + glossary
    st.sidebar.title("Modes")
    mode = st.sidebar.radio(
        "Choose mode:",
        ["Clinician", "Student", "Researcher", "Tour"],
        index=1,
    )
    explain_button("mode", "Explain modes")

    st.sidebar.markdown("---")
    st.sidebar.subheader("CiliaBot (educational only)")
    explain_button("cilia_bot", "What is CiliaBot?")
    user_q = st.sidebar.text_input("Ask about PCD / CBF / CBP / this app:")
    if user_q.strip():
        answer = ask_gpt(user_q)
        st.sidebar.write(answer)
        st.sidebar.caption("CiliaBot is for education only, not medical advice.")
        log_ciliabot_question(user_q, mode)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Feature glossary")
    feature_choice = st.sidebar.selectbox(
        "Pick a feature to learn about:",
        [
            "CBF",
            "CBP regularity score",
            "CBP amplitude",
            "Zero-crossing rate",
            "Variance",
            "Prediction",
            "Signals (raw vs filtered)",
            "FFT",
            "Simulation sliders",
            "Toy cilium view",
            "Multi-cilia visualizer",
        ],
    )

    glossary_map = {
        "CBF": "cbf",
        "CBP regularity score": "cbp_score",
        "CBP amplitude": "cbp_amp",
        "Zero-crossing rate": "zcr",
        "Variance": "var",
        "Prediction": "prediction",
        "Signals (raw vs filtered)": "signals",
        "FFT": "fft",
        "Simulation sliders": "sim_sliders",
        "Toy cilium view": "sim_cilium",
        "Multi-cilia visualizer": "multi_cilia",
    }
    st.sidebar.info(FEATURE_HELP[glossary_map[feature_choice]])

    # Guided tour state
    if "tour_step" not in st.session_state:
        st.session_state.tour_step = 0

    if mode == "Tour":
        step = st.session_state.tour_step
        st.markdown("### Guided tour")
        st.write(
            f"Step {step+1} of {len(TOUR_STEPS)}: **{TOUR_STEPS[step]['title']}**"
        )
        st.info(TOUR_STEPS[step]["body"])

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("Previous step", disabled=(step == 0)):
                st.session_state.tour_step = max(0, step - 1)
                st.experimental_rerun()
        with col_next:
            if st.button("Next step", disabled=(step == len(TOUR_STEPS) - 1)):
                st.session_state.tour_step = min(len(TOUR_STEPS) - 1, step + 1)
                st.experimental_rerun()

    # Train model
    try:
        model, feat_cols, acc, report_txt = train_model("labeled_cilia_video_dataset.csv")
        st.success(f"Random Forest trained. Test accuracy: **{acc*100:.1f}%**")
    except Exception as e:
        st.error(f"Could not train model: {e}")
        return

    if mode == "Researcher":
        with st.expander("Show training classification report"):
            st.text(report_txt)

        with st.expander("View CiliaBot question log (local)"):
            log_path = "cilia_bot_log.csv"
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                st.dataframe(log_df.tail(100))
            else:
                st.write("No CiliaBot questions have been logged yet in this environment.")

    # Tabs
    tab_analyse, tab_sim, tab_multi = st.tabs(
        ["Analyse video", "Simulation lab", "Multi-cilia visualizer"]
    )

    # ---------------- Tab: Analyse video ----------------
    with tab_analyse:
        st.subheader("Upload a cilia video")
        explain_button("signals", "Explain what we analyse")

        uploaded_video = st.file_uploader(
            "Upload a video file (.avi, .mp4, .mov)", type=["avi", "mp4", "mov"]
        )

        max_frames = st.slider(
            "Max frames to analyse",
            min_value=100,
            max_value=400,
            value=256,
            step=16,
        )

        if uploaded_video is None:
            st.info("Upload a video to see the analysis.")
        else:
            with st.spinner("Processing video and extracting features..."):
                try:
                    feats, fps, n_frames = extract_features_from_uploaded_video(
                        uploaded_video, max_frames=max_frames
                    )
                except Exception as e:
                    st.error(f"Error reading or processing video: {e}")
                    return

            st.write(f"**Video info:** {n_frames} frames at {fps:.1f} fps")

            # Prepare scalar features for display
            feats_for_display = {
                k: v for k, v in feats.items()
                if k not in ["raw_signal", "filtered_signal"]
            }

            if mode == "Clinician":
                st.markdown("### Key metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("CBF (Hz)", f"{feats_for_display['global_cbf_hz']:.2f}")
                col2.metric(
                    "Regularity", f"{feats_for_display['global_cbp_sine_score']:.2f}"
                )
                col3.metric(
                    "Amplitude", f"{feats_for_display['global_cbp_amp']:.3f}"
                )
                explain_button("cbf", "Explain CBF")
                explain_button("cbp_score", "Explain regularity score")
                explain_button("cbp_amp", "Explain amplitude")

            elif mode in ["Student", "Tour"]:
                st.markdown("### Extracted global features")
                st.table(pd.DataFrame([feats_for_display]))
                explain_button("cbf", "Explain CBF")
                explain_button("cbp_score", "Explain regularity score")
                explain_button("cbp_amp", "Explain amplitude")
                explain_button("zcr", "Explain zero-crossing rate")
                explain_button("var", "Explain variance")

            else:  # Researcher
                st.markdown("### Raw feature table")
                st.table(pd.DataFrame([feats_for_display]))

            # Prediction
            row = [feats_for_display.get(name, np.nan) for name in feat_cols]
            X_new = pd.DataFrame([row], columns=feat_cols)

            pred_class = model.predict(X_new)[0]
            pred_prob = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]
                class_to_idx = {c: i for i, c in enumerate(model.classes_)}
                pred_prob = float(proba[class_to_idx[pred_class]])

            st.subheader("Model prediction")
            explain_button("prediction", "Explain how this prediction works")

            if pred_class == "Healthy":
                msg = "Prediction: **Healthy-like cilia motion**"
                if pred_prob is not None:
                    msg += f" (approx. confidence {pred_prob*100:.1f}%)"
                st.success(msg)
            else:
                msg = "Prediction: **PCD-like cilia motion**"
                if pred_prob is not None:
                    msg += f" (approx. confidence {pred_prob*100:.1f}%)"
                st.error(msg)

            # Signal plots
            raw = feats.get("raw_signal", None)
            flt = feats.get("filtered_signal", None)
            if raw is not None and flt is not None:
                st.markdown("### Motion signal over time")
                explain_button("signals", "Explain these plots")
                t = np.arange(len(raw)) / float(fps)

                fig_sig, ax_sig = plt.subplots()
                ax_sig.plot(t, raw, label="Raw mean intensity")
                ax_sig.plot(t, flt, label="Filtered (2–30 Hz)")
                ax_sig.set_xlabel("Time (seconds)")
                ax_sig.set_ylabel("Signal")
                ax_sig.grid(True)
                ax_sig.legend()
                st.pyplot(fig_sig)

                if st.checkbox("Show FFT of filtered signal"):
                    explain_button("fft", "Explain the FFT view")
                    spec = np.abs(rfft(flt))
                    freqs = rfftfreq(len(flt), 1.0 / fps)
                    fig_fft, ax_fft = plt.subplots()
                    ax_fft.plot(freqs, spec)
                    ax_fft.set_xlim(0, 40)
                    ax_fft.set_xlabel("Frequency (Hz)")
                    ax_fft.set_ylabel("Magnitude")
                    ax_fft.set_title("FFT of filtered signal")
                    ax_fft.grid(True)
                    st.pyplot(fig_fft)

    # ---------------- Tab: Simulation lab ----------------
    with tab_sim:
        st.header("Cilia motion simulation lab")
        explain_button("sim_sliders", "Explain the simulation controls")

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
            st.write("Presets:")
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
                "- **Healthy-like**: higher CBF, regular beating, stronger amplitude\n"
                "- **PCD-like**: lower CBF, irregular pattern, weaker amplitude"
            )

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
        explain_button("sim_cilium", "Explain this toy cilium")

        idx = st.slider(
            "Pick a time index",
            min_value=0,
            max_value=len(sig_sim) - 1,
            value=len(sig_sim) // 2,
        )

        frame_val = sig_sim[idx]
        sig_std = np.std(sig_sim) if np.std(sig_sim) > 0 else 1.0
        norm_val = frame_val / (3.0 * sig_std)

        fig_cil = draw_cilium_frame(norm_val)
        st.pyplot(fig_cil)

    # ---------------- Tab: Multi-cilia visualizer ----------------
    with tab_multi:
        st.header("Multi-cilia visualizer")
        explain_button("multi_cilia", "Explain this visualizer")

        n = st.slider("Number of cilia", 5, 30, 12)
        cbf2 = st.slider("CBF (Hz) for visualizer", 2.0, 20.0, 10.0)

        t = np.linspace(0, 1, 200)
        fig_multi, ax_multi = plt.subplots(figsize=(8, 5))

        for i in range(n):
            phase = np.random.uniform(0, 2 * np.pi)
            y = np.sin(2 * np.pi * cbf2 * t + phase)
            ax_multi.plot(t, y + i * 2, linewidth=1)

        ax_multi.set_axis_off()
        st.pyplot(fig_multi)


if __name__ == "__main__":
    main()
