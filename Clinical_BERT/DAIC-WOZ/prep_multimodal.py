# prep_multimodal.py
import ast, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV = "merged_audio_text_transcripts.csv"
AUDIO_COLS = ["pitch_mean","energy"] + [f"mfcc_{i}" for i in range(13)]

def parse_labels(x):
    if isinstance(x, list): return x
    if isinstance(x, str) and x.strip():
        try: return ast.literal_eval(x)
        except Exception: return []
    return []

df = pd.read_csv(CSV)
df["emotion_labels"] = df["emotion_labels"].apply(parse_labels)
df = df[df["emotion_labels"].apply(len) > 0]
df["text"] = df["text"].astype(str)

# Build consistent label space (IMPORTANT: match your fine-tuned ClinicalBERT label set)
all_labels = sorted({lab for labs in df["emotion_labels"] for lab in labs})
label2id = {l:i for i,l in enumerate(all_labels)}
id2label = {i:l for l,i in label2id.items()}
num_labels = len(label2id)

def to_multi_hot(lst):
    v = [0]*num_labels
    for l in lst:
        if l in label2id: v[label2id[l]] = 1
    return v

df["label_vec"] = df["emotion_labels"].apply(to_multi_hot)

# Split by session_id to prevent speaker leakage
train_ids, temp_ids = train_test_split(df["session_id"].unique(), test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
is_in = lambda ids: df["session_id"].isin(ids)
train_df, val_df, test_df = df[is_in(train_ids)].copy(), df[is_in(val_ids)].copy(), df[is_in(test_ids)].copy()

# Standardize audio features using train only
scaler = StandardScaler()
train_df[AUDIO_COLS] = scaler.fit_transform(train_df[AUDIO_COLS])
val_df[AUDIO_COLS]   = scaler.transform(val_df[AUDIO_COLS])
test_df[AUDIO_COLS]  = scaler.transform(test_df[AUDIO_COLS])

# Save splits & artifacts
train_df.to_parquet("mm_train.parquet", index=False)
val_df.to_parquet("mm_val.parquet", index=False)
test_df.to_parquet("mm_test.parquet", index=False)
with open("mm_labels.json","w") as f: json.dump({"label2id":label2id,"id2label":id2label}, f)
import joblib; joblib.dump(scaler, "mm_audio_scaler.joblib")
print(f"Labels: {all_labels}")
