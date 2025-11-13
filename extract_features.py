import os
import argparse
import numpy as np
import cv2
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
def load_video_frames(video_path, max_frames=300):
    """
    Read up to max_frames grayscale frames from a video.
    Returns:
    frames_array: shape (T, H, W)
    fps: frames per second
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot Open video: " + str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_list.append(gray.astype(np.float32))
        frame_count += 1

        # stop if reached max_frames
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    if len(frames_list) == 0:
        raise RuntimeError("No frames read from " + str(video_path))

    frames_array = np.stack(frames_list, axis=0)
    return frames_array, fps


# ---------------------------------------------------
# BANDPASS FILTER
# ---------------------------------------------------

def bandpass_filter(signal, fs, low=2.0, high=30.0, order=2):
    """
    Simple Butterworth band-pass filter.
    Keeps frequencies between low and high (Hz).
    fs = sampling rate (frames per second)
    """
    nyquist = 0.5 * fs
    low_cut = low / nyquist
    high_cut = high / nyquist

    if high_cut >= 1.0:
        high_cut = 0.99

    b, a = butter(order, [low_cut, high_cut], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal



def fft_peak(signal, fs, fmin=2.0, fmax=30.0):
    """
    Find the biggest frequency and its magnitude in [fmin, fmax].
    signal: 1D array (time series)
    fs: sampling rate (Hz)
    """
    signal = np.array(signal, dtype=float)
    signal = signal - np.mean(signal)

    filtered = bandpass_filter(signal, fs, fmin, fmax)

    spectrum = rfft(filtered)
    freqs = rfftfreq(len(filtered), 1.0 / fs)
    mags = np.abs(spectrum)

    best_freq = np.nan
    best_mag = np.nan

    for i in range(len(freqs)):
        f = freqs[i]

        # skip frequencies outside our band
        if f < fmin or f > fmax:
            continue

        if np.isnan(best_mag) or mags[i] > best_mag:
            best_mag = mags[i]
            best_freq = f

    return float(best_freq), float(best_mag), filtered


# CBP: how sine-like the filtered signal is at main_freq
def cbp_sine_score(filtered_signal, fs, main_freq):
    """
    Estimate how 'sine-wave-like' the motion is at the main frequency.
    Returns a value around -1..1 (closer to 1 = clean, regular oscillation).
    """
    if filtered_signal is None:
        return np.nan
    if np.isnan(main_freq) or main_freq <= 0:
        return np.nan

    sig = np.array(filtered_signal, dtype=float)
    if sig.size == 0:
        return np.nan

    # time vector
    t = np.arange(len(sig)) / float(fs)
    ideal = np.sin(2.0 * np.pi * main_freq * t)

    # normalize
    sig = sig - np.mean(sig)
    sig_std = np.std(sig)
    ideal_std = np.std(ideal)
    if sig_std == 0 or ideal_std == 0:
        return np.nan

    sig = sig / sig_std
    ideal = ideal / ideal_std

    # simple correlation (cosine similarity-like)
    score = float(np.mean(sig * ideal))
    return score
def get_global_features(frames, fps):
    """
    Compute global motion features for the whole video.
    Uses the average brightness of each frame over time.
    """
    # mean brightness per frame: shape (T,)
    mean_signal = frames.mean(axis=(1, 2))

    # main frequency (CBF) + its amplitude + filtered signal
    cbf, amp, filtered = fft_peak(mean_signal, fps)

    # variance and zero-crossing rate on filtered signal
    try:
        var_value = float(np.var(filtered))
        sign_changes = np.diff(np.sign(filtered)) != 0
        zcr_value = float(np.mean(sign_changes))
    except Exception:
        var_value = np.nan
        zcr_value = np.nan

    # CBP-like measures: how sine-like and how big the oscillation is
    cbp_score = cbp_sine_score(filtered, fps, cbf)
    cbp_amp = float(np.std(filtered)) if not np.isnan(cbf) else np.nan

    result = {
        "global_cbf_hz": cbf,
        "global_peak_amp": amp,
        "global_var": var_value,
        "global_zcr": zcr_value,
        "global_cbp_sine_score": cbp_score,
        "global_cbp_amp": cbp_amp,
    }

    return result
def get_tile_features(frames, fps, tile_size=32, fmin=2.0, fmax=30.0):
    """
    Compute local motion features for each tile (tile_size x tile_size).
    """
    T, H, W = frames.shape
    tile_feature_list = []

    # loop over tiles
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):

            # get tile over all frames: (T, tile_size, tile_size)
            tile = frames[:, y:y + tile_size, x:x + tile_size]

            # skip empty tiles (can happen at borders)
            if tile.size == 0:
                continue

            # average brightness of tile over time
            sig = tile.mean(axis=(1, 2))
            sig = sig - np.mean(sig)

            try:
                # fft_peak now returns cbf, amp, filtered
                cbf, amp, filtered = fft_peak(sig, fps, fmin, fmax)

                # if fft_peak failed
                if np.isnan(cbf):
                    continue

                var_value = float(np.var(filtered))
                sign_changes = np.diff(np.sign(filtered)) != 0
                zcr_value = float(np.mean(sign_changes))

                # CBP features
                cbp_score = cbp_sine_score(filtered, fps, cbf)
                cbp_amp = float(np.std(filtered))

                tile_info = {
                    "x": x,
                    "y": y,
                    "tile_cbf_hz": cbf,
                    "tile_peak_amp": amp,
                    "tile_var": var_value,
                    "tile_zcr": zcr_value,
                    "tile_cbp_sine_score": cbp_score,
                    "tile_cbp_amp": cbp_amp,
                }

                tile_feature_list.append(tile_info)

            except Exception:
                # if something goes wrong for this tile, just skip it
                continue

    return tile_feature_list
def analyze_video(video_path, label, tile_size=32, max_frames=300,
                fmin=2.0, fmax=30.0):
    print("")
    print(">>> Analyzing:", video_path, "Label:", label)

    frames, fps = load_video_frames(video_path, max_frames)
    print("Loaded", frames.shape[0], "frames at", round(fps, 1), "fps")

    # global features for whole video
    global_feats = get_global_features(frames, fps)

    # tile-based features
    tile_feats = get_tile_features(frames, fps, tile_size, fmin, fmax)

    # attach video info to each tile row
    for row in tile_feats:
        row["video"] = os.path.basename(video_path)
        row["label"] = label
        row["fps"] = fps

    # now compute summary from tile CBF values
    if len(tile_feats) > 0:
        cbf_list = [row["tile_cbf_hz"] for row in tile_feats]
        cbf_array = np.array(cbf_list, dtype=float)

        cbf_mean = float(np.mean(cbf_array))
        cbf_std = float(np.std(cbf_array))
        cbf_min = float(np.min(cbf_array))
        cbf_max = float(np.max(cbf_array))

        # fractions in different CBF ranges
        frac_low = float(np.mean(cbf_array < 3.0))
        frac_normal = float(np.mean((cbf_array >= 5.0) & (cbf_array <= 20.0)))
        frac_high = float(np.mean(cbf_array > 20.0))

        global_feats["cbf_mean_tiles"] = cbf_mean
        global_feats["cbf_std_tiles"] = cbf_std
        global_feats["cbf_min_tiles"] = cbf_min
        global_feats["cbf_max_tiles"] = cbf_max
        global_feats["frac_low_cbf"] = frac_low
        global_feats["frac_normal_cbf"] = frac_normal
        global_feats["frac_high_cbf"] = frac_high
        global_feats["n_tiles_used"] = int(len(cbf_array))

    else:
        # no tiles used
        global_feats["cbf_mean_tiles"] = np.nan
        global_feats["cbf_std_tiles"] = np.nan
        global_feats["cbf_min_tiles"] = np.nan
        global_feats["cbf_max_tiles"] = np.nan
        global_feats["frac_low_cbf"] = np.nan
        global_feats["frac_normal_cbf"] = np.nan
        global_feats["frac_high_cbf"] = np.nan
        global_feats["n_tiles_used"] = 0

    # add meta info
    global_feats["video"] = os.path.basename(video_path)
    global_feats["label"] = label
    global_feats["fps"] = fps

    return global_feats, tile_feats
def label_from_name(name):
    """
    Simple rule:
    if "Healthy" in the label -> class = "Healthy"
    else                       -> class = "PCD"
    """
    if "Healthy" in name:
        return "Healthy"
    else:
        return "PCD"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract global and tile-based features from cilia videos."
    )
    parser.add_argument("--base_dir", help="Directory containing video files (default: current working dir)",
                        default=os.getcwd())
    parser.add_argument("--max_frames", type=int, default=256, help="Maximum frames to read per video")
    parser.add_argument("--tile_size", type=int, default=32, help="Tile size in pixels")
    parser.add_argument("--output_dir", default=".", help="Directory to write CSV outputs")

    args = parser.parse_args()

    base_dir = args.base_dir
    max_frames = args.max_frames
    tile_size = args.tile_size
    output_dir = args.output_dir

    video_label_map = {
        "inline-supplementary-material-1.avi": "PCD_DNAH5",
        "inline-supplementary-material-2.avi": "PCD_ZMYND10",
        "inline-supplementary-material-3.avi": "PCD_CCDC40",
        "inline-supplementary-material-4.avi": "PCD_DNAH11",
        "inline-supplementary-material-5.avi": "PCD_CCDC151",
        "inline-supplementary-material-6.avi": "Healthy_side",
        "inline-supplementary-material-7.avi": "Healthy_top",
    }

    all_video_rows = []
    all_tile_rows = []

    for filename, label in video_label_map.items():
        full_path = os.path.join(base_dir, filename)

        if not os.path.exists(full_path):
            print("Warning: missing file", full_path, "- skipping")
            continue

        video_row, tile_rows = analyze_video(
            full_path,
            label,
            tile_size=tile_size,
            max_frames=max_frames
        )

        all_video_rows.append(video_row)
        all_tile_rows.extend(tile_rows)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # save per-video CSV
    if len(all_video_rows) > 0:
        df_videos = pd.DataFrame(all_video_rows)
        df_videos.to_csv(os.path.join(output_dir, "cbf_per_video.csv"), index=False)
        print("")
        print("Saved cbf_per_video.csv with", len(df_videos), "videos to", output_dir)

    # save per-tile CSV
    if len(all_tile_rows) > 0:
        df_tiles = pd.DataFrame(all_tile_rows)
        df_tiles.to_csv(os.path.join(output_dir, "cbf_per_tile.csv"), index=False)
        print("Saved cbf_per_tile.csv with", len(df_tiles), "tiles to", output_dir)

    # add class labels (Healthy / PCD) and save labeled CSVs
    if len(all_video_rows) > 0:
        df_videos["class"] = df_videos["label"].apply(label_from_name)
        df_videos.to_csv(os.path.join(output_dir, "labeled_cilia_video_dataset.csv"), index=False)
        print("Saved labeled_cilia_video_dataset.csv to", output_dir)

    if len(all_tile_rows) > 0:
        df_tiles["class"] = df_tiles["label"].apply(label_from_name)
        df_tiles.to_csv(os.path.join(output_dir, "labeled_cilia_tile_dataset.csv"), index=False)
        print("Saved labeled_cilia_tile_dataset.csv to", output_dir)
