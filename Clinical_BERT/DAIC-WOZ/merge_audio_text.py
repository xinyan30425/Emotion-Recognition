# 6. merge_audio_text.py
import pandas as pd
import numpy as np
import os

LABELS_WITH_IDS = "labeled_transcripts_with_ids_validated.csv"
AUDIO_FEATS     = "processed/features/audio_features.csv"
OUTPUT_MERGED   = "merged_audio_text_transcripts.csv"

df_text = pd.read_csv(LABELS_WITH_IDS)
df_audio = pd.read_csv(AUDIO_FEATS)

# Expand mfcc_mean list into columns
def expand_mfcc(df):
    # handle stringified lists
    def parse(v):
        if isinstance(v, str):
            try:
                return eval(v)
            except Exception:
                return []
        return v

    mfcc_lists = df["mfcc_mean"].apply(parse)
    max_len = mfcc_lists.map(lambda x: len(x) if isinstance(x, list) else 0).max()

    for i in range(max_len):
        df[f"mfcc_{i}"] = mfcc_lists.apply(lambda x: float(x[i]) if isinstance(x, list) and len(x) > i else np.nan)

    return df.drop(columns=["mfcc_mean"])

df_audio = expand_mfcc(df_audio)

# Merge on id
df_merged = pd.merge(df_text, df_audio, on="id", how="inner")

df_merged.to_csv(OUTPUT_MERGED, index=False)
print(f"Merged {len(df_merged)} rows → {OUTPUT_MERGED}")
