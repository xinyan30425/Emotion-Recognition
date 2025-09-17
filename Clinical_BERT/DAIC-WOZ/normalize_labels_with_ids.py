# 3. normalize_labels_with_ids.py
import pandas as pd
import os

labels_path = "labeled_transcripts_emotions.csv"
out_path = "labeled_transcripts_with_ids.csv"

df = pd.read_csv(labels_path)

# sort per session for deterministic order
df = df.sort_values(["session_id", "start_time", "stop_time"]).reset_index(drop=True)

# local_idx per session, matching the segmenter ordering
df["local_idx"] = df.groupby("session_id").cumcount()

# build filename id to match audio clips and features
df["id"] = df.apply(lambda r: f"{r['session_id']}_Participant_{int(r['local_idx'])}.wav", axis=1)

# (optional) keep only rows with at least one label
def has_label(x):
    try:
        arr = eval(x) if isinstance(x, str) else []
        return isinstance(arr, (list, tuple)) and len(arr) > 0
    except Exception:
        return False

df_labeled = df[df["emotion_labels"].apply(has_label)].copy()

df_labeled.to_csv(out_path, index=False)
print(f"Wrote {len(df_labeled)} labeled rows with ids {out_path}")
