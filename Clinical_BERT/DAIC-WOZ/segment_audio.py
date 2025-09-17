# 4. segment_audio.py
from pydub import AudioSegment
import pandas as pd
import os

def segment_audio(session_id, audio_path, transcript_path, out_dir, seg_manifest_dir):
    audio = AudioSegment.from_wav(audio_path)
    df = pd.read_csv(transcript_path, delimiter="\t")

    # Participant-only, stable order
    df = df[df["speaker"].str.lower() == "participant"].copy()
    df = df.sort_values(["start_time", "stop_time"]).reset_index(drop=True)
    df["local_idx"] = df.groupby(lambda _: session_id).cumcount()

    rows = []
    for _, row in df.iterrows():
        start_ms = int(row["start_time"] * 1000)
        stop_ms  = int(row["stop_time"] * 1000)
        local_idx = int(row["local_idx"])
        fname = f"{session_id}_Participant_{local_idx}.wav"

        # export
        clip_path = os.path.join(out_dir, fname)
        audio[start_ms:stop_ms].export(clip_path, format="wav")

        rows.append({
            "session_id": session_id,
            "local_idx": local_idx,
            "start_time": row["start_time"],
            "stop_time": row["stop_time"],
            "filename": fname
        })

    # write manifest for this session
    os.makedirs(seg_manifest_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(seg_manifest_dir, f"{session_id}.csv"), index=False)
    print(f"Segmented {session_id}: {len(rows)} Participant clips")

if __name__ == "__main__":
    data_dir = "data"
    clips_root = "processed/clips"
    seg_manifest_dir = "processed/segments"
    os.makedirs(clips_root, exist_ok=True)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        session_id = folder  # e.g., "300_P"
        audio_path = os.path.join(folder_path, f"{folder[:3]}_AUDIO.wav")
        transcript_path = os.path.join(folder_path, f"{folder[:3]}_TRANSCRIPT.csv")

        if not (os.path.exists(audio_path) and os.path.exists(transcript_path)):
            print(f"⚠️ Missing files in {folder}, skipping...")
            continue

        out_dir = os.path.join(clips_root, session_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f" Segmenting {session_id} (Participant only)…")
        segment_audio(session_id, audio_path, transcript_path, out_dir, seg_manifest_dir)
