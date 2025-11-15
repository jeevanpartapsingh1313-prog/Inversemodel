import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


FEATURE_HELP = {
    "cbf": {
        "Student": (
            "CBF means Ciliary Beat Frequency. It tells you how many times the cilia beat "
            "every second. We estimate this from the brightness pattern and find the strongest "
            "repeating wave using the FFT."
        ),
        "Clinician": (
            "Ciliary Beat Frequency is computed by band-pass filtering the mean pixel-intensity "
            "trace (2–30 Hz) and identifying the dominant spectral peak in the FFT magnitude. "
            "It approximates the primary ciliary stroke frequency after removing low-frequency "
            "drift and high-frequency noise."
        ),
    },
    "cbp_score": {
        "Student": (
            "This regularity score tells you how smooth and sine-like the motion is. Values near 1 "
            "mean very regular beating; lower values mean more irregular motion."
        ),
        "Clinician": (
            "The CBP regularity score is the normalized correlation between the filtered trace and "
            "an ideal sinusoid at the dominant frequency. Low scores suggest dyskinetic or "
            "asynchronous beating patterns typical of PCD."
        ),
    },
    "cbp_amp": {
        "Student": (
            "Amplitude shows how strong the cilia movement is. Higher values mean the signal has "
            "bigger swings; lower values mean weaker motion."
        ),
        "Clinician": (
            "CBP amplitude is approximated by the standard deviation of the band-passed signal. "
            "It reflects stroke displacement magnitude; reduced amplitude is associated with "
            "hypokinetic or nearly static cilia."
        ),
    },
    "zcr": {
        "Student": (
            "Zero-crossing rate counts how often the signal crosses zero. A higher rate can mean "
            "faster or more jittery changes."
        ),
        "Clinician": (
            "Zero-crossing rate is a simple descriptor of oscillatory irregularity. Very high values "
            "can indicate noisy or fragmented motion rather than a clean periodic beat."
        ),
    },
    "var": {
        "Student": (
            "Variance tells you how much the signal values change over time. Bigger variance means "
            "larger ups and downs; small variance means the signal stays flatter."
        ),
        "Clinician": (
            "Variance of the filtered trace reflects total oscillatory energy. Very low variance can "
            "indicate near-static cilia; high variance reflects strong, sustained motion."
        ),
    },
    "signals": {
        "Student": (
            "The raw signal is the average brightness of each frame. The filtered signal removes slow "
            "drifts and noise so you can see a cleaner repeating wave from the cilia motion."
        ),
        "Clinician": (
            "The raw signal is the frame-wise mean grayscale intensity. After applying a Butterworth "
            "band-pass filter (2–30 Hz), the filtered trace highlights physiologic ciliary motion by "
            "suppressing low-frequency camera drift and high-frequency noise."
        ),
    },
    "fft": {
        "Student": (
            "The FFT plot shows which frequencies are strong in the motion signal. The tallest peak is "
            "usually the main beat frequency of the cilia."
        ),
        "Clinician": (
            "The FFT magnitude spectrum is computed using the real-valued rFFT. Peaks in the expected "
            "physiologic range correspond to dominant beating frequencies, while broadened or weak peaks "
            "can indicate dyskinetic motion or poor signal quality."
        ),
    },
    "sim_sliders": {
        "Student": (
            "These controls let you play with a fake cilia signal. You can change how fast it beats, how "
            "regular it is, how strong the movement is, and how long it runs."
        ),
        "Clinician": (
            "The simulation exposes controllable parameters for synthetic oscillatory signals: frequency, "
            "regularity via noise and envelope modulation, amplitude, and duration. It is intended for "
            "intuition-building, not biophysical modelling."
        ),
    },
    "sim_cilium": {
        "Student": (
            "This view draws a simple line that tilts according to the signal value, like a cartoon cilium "
            "moving back and forth."
        ),
        "Clinician": (
            "The toy cilium view maps a normalized signal sample to an angular displacement. It is a didactic "
            "visual and does not model full 3D axonemal mechanics."
        ),
    },
    "prediction": {
        "Student": (
            "The model looks at the numbers above and guesses whether the motion looks more like healthy cilia "
            "or PCD-like cilia. It is only for learning, not diagnosis."
        ),
        "Clinician": (
            "The Random Forest operates on six global features to assign labels such as Healthy-like or PCD-like. "
            "This is a research and teaching prototype and is not validated for clinical decision-making."
        ),
    },
    "cilia_bot": {
        "Student": (
            "CiliaBot is a chat assistant that explains PCD, motile cilia, CBF, CBP and what the graphs mean in "
            "simple language. It is only for education, not for medical advice."
        ),
        "Clinician": (
            "CiliaBot uses a large language model to provide educational explanations about ciliary physiology, "
            "PCD, and this app. It does not provide diagnostic guidance or replace clinical judgement."
        ),
    },
}


def explain_button(widget_key: str, label: str = "Explain", feature_key: str = None, mode: str = None):
    if st.button(label, key=f"explain_{widget_key}"):
        key = feature_key or widget_key
        info = FEATURE_HELP.get(key)
        if isinstance(info, dict) and mode is not None:
            st.info(info.get(mode, "No explanation available for this mode."))
        elif isinstance(info, str):
            st.info(info)
        else:
            st.info("No explanation available for this item.")


def log_ciliabot_question(question: str, mode: str):
    question = (question or "").strip()
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


def ask_gpt(chat_history):
    api_key = None
    if "openrouter_api_key" in st.secrets:
        api_key = st.secrets["openrouter_api_key"]
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return "⚠️ Missing OpenRouter API key."

    messages = [
        {
            "role": "system",
            "content": (
                "You are CiliaBot, a friendly explainer for clinicians and students. "
                "Explain primary ciliary dyskinesia (PCD), motile cilia, ciliary beat frequency (CBF), "
                "ciliary beat pattern (CBP), and this app's features in clear, accurate language. "
                "You provide education only and never medical advice or diagnosis."
            ),
        }
    ] + chat_history

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "CiliaMotionApp",
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct",
                "messages": messages,
                "max_tokens": 400,
            },
            timeout=30,
        )
        data = resp.json()

        if "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            return f"⚠️ OpenRouter API error: {msg}"

        choices = data.get("choices")
        if not choices:
            return f"⚠️ Unexpected OpenRouter response: {data}"

        return choices[0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error contacting OpenRouter: {e}"


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
    cbp_score_val = cbp_sine_score(filtered, fps, cbf)
    cbp_amp_val = float(np.std(filtered))

    return {
        "global_cbf_hz": cbf,
        "global_peak_amp": amp,
        "global_var": var,
        "global_zcr": zcr,
        "global_cbp_sine_score": cbp_score_val,
        "global_cbp_amp": cbp_amp_val,
        "raw_signal": mean_signal,
        "filtered_signal": filtered,
    }


def load_frames_from_file(path, max_frames=300):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
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
    if fps <= 0:
        fps = 60.0
    return frames, fps


def extract_features_from_uploaded_video(uploaded_file, max_frames=300):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    tmp.write(uploaded_file.read())
    tmp.close()

    frames, fps = load_frames_from_file(tmp.name, max_frames=max_frames)
    feats = compute_global_features(frames, fps)
    return feats, fps, frames.shape[0]


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


@st.cache_data
def load_healthy_baseline(csv_path="labeled_cilia_video_dataset.csv", feat_cols=None):
    df = load_training_data(csv_path)

    if feat_cols is None:
        feat_cols = [
            "global_cbf_hz",
            "global_peak_amp",
            "global_var",
            "global_zcr",
            "global_cbp_sine_score",
            "global_cbp_amp",
        ]

    df_clean = df.dropna(subset=feat_cols + ["class"])
    healthy = df_clean[df_clean["class"] == "Healthy"]

    if healthy.empty:
        return None

    mean = healthy[feat_cols].mean()
    std = healthy[feat_cols].std()

    baseline = pd.DataFrame(
        {
            "feature": feat_cols,
            "Healthy mean": mean.values,
            "Healthy low (mean-1σ)": (mean - std).values,
            "Healthy high (mean+1σ)": (mean + std).values,
        }
    )

    return baseline


def main():
    st.set_page_config(page_title="Cilia Motion Classifier Pro", layout="wide")
    st.title("🧬 Cilia Motion Classifier & Simulation Lab")

    st.sidebar.title("Modes")
    mode = st.sidebar.radio(
        "Choose mode:",
        ["Clinician", "Student"],
        index=1,
    )

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
            "CiliaBot",
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
        "CiliaBot": "cilia_bot",
    }
    gkey = glossary_map[feature_choice]
    ginfo = FEATURE_HELP.get(gkey)
    if isinstance(ginfo, dict):
        st.sidebar.info(ginfo.get(mode, "No explanation available for this mode."))
    elif isinstance(ginfo, str):
        st.sidebar.info(ginfo)
    else:
        st.sidebar.info("No explanation available.")

    try:
        model, feat_cols, acc, report_txt = train_model("labeled_cilia_video_dataset.csv")
        st.success(f"Random Forest trained. Test accuracy: {acc*100:.1f}%")
    except Exception as e:
        st.error(f"Could not train model: {e}")
        return

    healthy_baseline = load_healthy_baseline("labeled_cilia_video_dataset.csv", feat_cols)

    tab_analyse, tab_sim, tab_chat = st.tabs(
        ["Analyse video", "Simulation lab", "Ask CiliaBot"]
    )

    with tab_analyse:
        st.subheader("Upload a cilia video")
        explain_button("video_upload", "Explain what we analyse", feature_key="signals", mode=mode)

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

            st.write(f"Video info: {n_frames} frames at {fps:.1f} fps")

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
                explain_button("cbf_metrics", "Explain CBF", feature_key="cbf", mode=mode)
                explain_button("cbp_score_metrics", "Explain regularity", feature_key="cbp_score", mode=mode)
                explain_button("cbp_amp_metrics", "Explain amplitude", feature_key="cbp_amp", mode=mode)

            elif mode == "Student":
                st.markdown("### Extracted global features")
                st.table(pd.DataFrame([feats_for_display]))
                explain_button("cbf_table", "Explain CBF", feature_key="cbf", mode=mode)
                explain_button("cbp_score_table", "Explain regularity score", feature_key="cbp_score", mode=mode)
                explain_button("cbp_amp_table", "Explain amplitude", feature_key="cbp_amp", mode=mode)
                explain_button("zcr_table", "Explain zero-crossing rate", feature_key="zcr", mode=mode)
                explain_button("var_table", "Explain variance", feature_key="var", mode=mode)

            if healthy_baseline is not None:
                this_video_values = []
                for fname in healthy_baseline["feature"]:
                    this_video_values.append(feats_for_display.get(fname, np.nan))

                compare_df = healthy_baseline.copy()
                compare_df["This video"] = this_video_values

                st.markdown("### How this video compares to healthy baseline")

                st.dataframe(
                    compare_df.style.format(
                        {
                            "Healthy mean": "{:.3f}",
                            "Healthy low (mean-1σ)": "{:.3f}",
                            "Healthy high (mean+1σ)": "{:.3f}",
                            "This video": "{:.3f}",
                        }
                    )
                )

                if mode == "Student":
                    st.caption(
                        "These values come from videos labeled Healthy in the training data. "
                        "Healthy mean and mean ± 1σ show a typical range for normal cilia. "
                        "The last column shows this video. This is for learning only."
                    )
                else:
                    st.caption(
                        "Baseline distributions are derived from the Healthy subset in "
                        "labeled_cilia_video_dataset.csv as mean ± 1 SD for each feature. "
                        "They are dataset-based references, not diagnostic thresholds."
                    )

            row = [feats_for_display.get(name, np.nan) for name in feat_cols]
            X_new = pd.DataFrame([row], columns=feat_cols)

            pred_class = model.predict(X_new)[0]
            pred_prob = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]
                class_to_idx = {c: i for i, c in enumerate(model.classes_)}
                pred_prob = float(proba[class_to_idx[pred_class]])

            st.subheader("Model prediction")
            explain_button("prediction_btn", "Explain how this prediction works", feature_key="prediction", mode=mode)

            if pred_class == "Healthy":
                msg = "Prediction: Healthy-like cilia motion"
                if pred_prob is not None:
                    msg += f" (approx. confidence {pred_prob*100:.1f}%)"
                st.success(msg)
            else:
                msg = "Prediction: PCD-like cilia motion"
                if pred_prob is not None:
                    msg += f" (approx. confidence {pred_prob*100:.1f}%)"
                st.error(msg)

            raw = feats.get("raw_signal", None)
            flt = feats.get("filtered_signal", None)
            if raw is not None and flt is not None:
                st.markdown("### Motion signal over time")
                explain_button("signals_plots", "Explain these plots", feature_key="signals", mode=mode)
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
                    explain_button("fft_btn", "Explain the FFT view", feature_key="fft", mode=mode)
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

    with tab_sim:
        st.header("Cilia motion simulation lab")
        explain_button("sim_sliders_btn", "Explain the simulation controls", feature_key="sim_sliders", mode=mode)

        st.markdown("""
        ### What this Simulation Lab does

        This section lets you explore how different ciliary motion characteristics affect the signal we measure.
        It does not use real videos. Instead, it generates a synthetic motion pattern so you can build intuition about the features used in analysis.

        You can adjust:

        • CBF (Ciliary Beat Frequency): how fast the cilia beat  
        • Regularity: how smooth or irregular the motion is  
        • Amplitude: how strong the beating is  
        • Duration: how long the simulated signal lasts  

        The plots update instantly so you can see how each parameter shapes the signal.  
        The toy cilium view shows a simple physical interpretation of the signal at a chosen moment.
        """)

        col1, col2 = st.columns(2)

        with col1:
            sim_cbf = st.slider(
                "Simulated CBF (beats per second)",
                min_value=2.0,
                max_value=20.0,
                value=8.0,
                step=0.5,
            )
            explain_button("sim_cbf", "Explain CBF slider", feature_key="cbf", mode=mode)

            sim_reg = st.slider(
                "Regularity (0 = very irregular, 1 = very smooth)",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
            )
            explain_button("sim_reg", "Explain regularity slider", feature_key="cbp_score", mode=mode)

            sim_amp = st.slider(
                "Motion amplitude",
                min_value=0.2,
                max_value=2.0,
                value=1.0,
                step=0.1,
            )
            explain_button("sim_amp", "Explain amplitude slider", feature_key="cbp_amp", mode=mode)

            sim_duration = st.slider(
                "Simulation duration (seconds)",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.5,
            )
            explain_button("sim_duration", "Explain duration", feature_key="sim_sliders", mode=mode)

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
                "- Healthy-like: higher CBF, regular beating, stronger amplitude\n"
                "- PCD-like: lower CBF, irregular pattern, weaker amplitude"
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
        explain_button("sim_signal", "Explain simulated signal", feature_key="signals", mode=mode)
        fig_sim, ax_sim = plt.subplots()
        ax_sim.plot(t_sim, sig_sim)
        ax_sim.set_xlabel("Time (seconds)")
        ax_sim.set_ylabel("Simulated signal")
        ax_sim.grid(True)
        st.pyplot(fig_sim)

        st.subheader("Toy cilium view (single frame)")
        explain_button("sim_cilium_btn", "Explain toy cilium", feature_key="sim_cilium", mode=mode)

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

    with tab_chat:
        st.header("CiliaBot: educational cilia explainer")
        explain_button("cilia_chat", "Explain how CiliaBot works", feature_key="cilia_bot", mode=mode)

        st.caption(
            "CiliaBot explains PCD, motile cilia, CBF, CBP and this app in simple language. "
            "It is for education only and does not give medical advice or diagnosis."
        )

        if "cilia_chat" not in st.session_state:
            st.session_state.cilia_chat = []

        for msg in st.session_state.cilia_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask CiliaBot about cilia, PCD, CBF, CBP, or this app")
        if user_msg:
            st.session_state.cilia_chat.append({"role": "user", "content": user_msg})
            log_ciliabot_question(user_msg, mode)
            with st.chat_message("assistant"):
                reply = ask_gpt(st.session_state.cilia_chat)
                st.markdown(reply)
            st.session_state.cilia_chat.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
