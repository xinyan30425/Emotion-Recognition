# 5. extract_audio_features.py
import os
import re
import pandas as pd
import numpy as np
import librosa

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    energy = float(np.sqrt(np.mean(y**2)))
    return {
        "mfcc_mean": [float(x) for x in mfcc.mean(axis=1)],
        "pitch_mean": float(np.nanmean(pitch)),
        "energy": energy,
    }

def clean_clip_name(base, clip):
    """
    Ensure clip filename starts with 'Participant_' only.
    Removes duplicated session prefixes like '300_P_' or '300_' at the start.
    """
    # remove leading "<base>_" if present (e.g., "300_P_")
    if clip.startswith(base + "_"):
        clip = clip[len(base) + 1 :]

    # also remove leading "<session_id>_" if present (e.g., "300_")
    session_id = base.split("_")[0]
    if clip.startswith(session_id + "_"):
        clip = clip[len(session_id) + 1 :]

    # as a last resort, strip any "<digits>_P_" or "<digits>_" prefixes
    clip = re.sub(r"^\d+_P_", "", clip)
    clip = re.sub(r"^\d+_", "", clip)

    return clip

if __name__ == "__main__":
    input_dir = "processed/clips"
    output_csv = "processed/features/audio_features.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []
    total = 0

    for base in sorted(os.listdir(input_dir)):
        base_path = os.path.join(input_dir, base)
        if not os.path.isdir(base_path):
            continue

        session_id = base.split("_")[0]  # "300_P" -> "300"

        for clip in sorted(os.listdir(base_path)):
            if not clip.endswith(".wav"):
                continue

            clip_clean = clean_clip_name(base, clip)          # e.g., "Participant_20.wav"
            final_id = f"{session_id}_{clip_clean}"           # e.g., "300_Participant_20.wav"

            feats = extract_features(os.path.join(base_path, clip))
            feats["id"] = final_id
            rows.append(feats)
            total += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Extracted features for {total} clips → {output_csv}")
    if not df.empty:
        print(" Example IDs:", list(df["id"].head(5)))
