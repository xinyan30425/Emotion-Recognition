# eval_late_fusion.py
# Evaluate trained late-fusion system: text model (HF), audio model (MLP), and fusion (prob average).
# - Loads thresholds if available; otherwise tunes thresholds on the validation set.
# - Reports micro/macro F1 and per-label precision/recall/F1 with supports.
# - Saves detailed CSVs and predictions.

import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# -----------------------
# Defaults (override via CLI)
# -----------------------
DEFAULT_TEXT_DIR   = "mm_late_fusion/text/best_model"
DEFAULT_AUDIO_PT   = "mm_late_fusion/audio/best_model.pt"
DEFAULT_THRESH_JSON= "mm_late_fusion/thresholds.json"
DEFAULT_LABEL_META = "mm_labels.json"

DEFAULT_TRAIN = "mm_train.parquet"   # (optional; not needed unless you want per-label supports)
DEFAULT_VAL   = "mm_val.parquet"
DEFAULT_TEST  = "mm_test.parquet"

AUDIO_COLS = ["pitch_mean","energy"] + [f"mfcc_{i}" for i in range(13)]
MAX_LEN    = 256
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Audio model (must match training architecture)
# -----------------------
class AudioClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )
    def forward(self, audio):
        x = audio.float()
        logits = self.net(x)
        return logits

# -----------------------
# Utilities
# -----------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def ensure_float_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in list(x)]
    return [float(x)]

def to_hf_text(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({
        "text": df["text"].tolist(),
        "labels": df["label_vec"].apply(ensure_float_list).tolist(),
    }))

def to_hf_audio(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({
        "audio": df[AUDIO_COLS].values.tolist(),
        "labels": df["label_vec"].apply(ensure_float_list).tolist(),
    }))

def collate_text(tokenizer):
    def _fn(batch):
        texts  = [ex["text"] for ex in batch]
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.float32)
        tok = tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        tok["labels"] = labels
        return tok
    return _fn

def collate_audio(batch):
    audio  = torch.tensor([ex["audio"] for ex in batch], dtype=torch.float32)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.float32)
    return {"audio": audio, "labels": labels}

def run_text_model(model, tokenizer, ds: Dataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_text(tokenizer))
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)
    return np.vstack(all_logits), np.vstack(all_labels)

def run_audio_model(model, ds: Dataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_audio)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].numpy()
            audio  = batch["audio"].to(device)
            logits = model(audio).detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)
    return np.vstack(all_logits), np.vstack(all_labels)

def tune_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-class threshold sweep in [0.1, 0.9] to maximize per-class F1."""
    C = probs.shape[1]
    thresholds = np.full(C, 0.5, dtype=np.float32)
    for i in range(C):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 17):
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds

def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds[None, :]).astype(int)

def metrics_report(y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict[int,str]) -> Tuple[Dict[str,float], pd.DataFrame]:
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_micro  = recall_score(y_true, y_pred, average="micro", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # per-label
    C = y_true.shape[1]
    rows = []
    for c in range(C):
        name  = id2label.get(c, f"label_{c}")
        prec  = precision_score(y_true[:,c], y_pred[:,c], zero_division=0)
        rec   = recall_score(y_true[:,c], y_pred[:,c], zero_division=0)
        f1    = f1_score(y_true[:,c], y_pred[:,c], zero_division=0)
        supp  = int(y_true[:,c].sum())
        rows.append({"label_id": c, "label_name": name, "precision": prec, "recall": rec, "f1": f1, "support": supp})
    df = pd.DataFrame(rows).sort_values("label_id")

    agg = {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "micro_precision": float(prec_micro),
        "micro_recall": float(rec_micro),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
    }
    return agg, df

def safe_load_thresholds(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
                return data
            except Exception:
                return {}
    return {}

def ensure_dirs():
    os.makedirs("eval_reports", exist_ok=True)
    os.makedirs("eval_predictions", exist_ok=True)

# -----------------------
# Main evaluation
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate late-fusion models (text, audio, fusion).")
    parser.add_argument("--text_dir", default=DEFAULT_TEXT_DIR, help="Path to saved text model dir")
    parser.add_argument("--audio_pt", default=DEFAULT_AUDIO_PT, help="Path to saved audio model state_dict (.pt)")
    parser.add_argument("--thresholds_json", default=DEFAULT_THRESH_JSON, help="Path to thresholds JSON")
    parser.add_argument("--label_meta", default=DEFAULT_LABEL_META, help="Path to mm_labels.json")
    parser.add_argument("--val_path", default=DEFAULT_VAL, help="Validation parquet (for tuning thresholds / alpha)")
    parser.add_argument("--test_path", default=DEFAULT_TEST, help="Test parquet")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--alpha_grid", type=str, default="0.0,0.25,0.5,0.75,1.0",
                        help="Comma-separated alpha values for fusion: p = alpha*text + (1-alpha)*audio")
    args = parser.parse_args()

    ensure_dirs()

    # Load label meta
    meta = json.load(open(args.label_meta, "r"))
    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    num_labels = len(label2id)

    # Load data
    val_df  = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    # Sanity checks
    for df, name in [(val_df, "val"), (test_df, "test")]:
        if "text" not in df.columns:
            raise ValueError(f"{name} missing 'text'")
        if "label_vec" not in df.columns:
            raise ValueError(f"{name} missing 'label_vec'")
        if any(c not in df.columns for c in AUDIO_COLS):
            missing = [c for c in AUDIO_COLS if c not in df.columns]
            raise ValueError(f"{name} missing audio cols: {missing}")
        # label lengths
        bad = df["label_vec"].apply(lambda x: len(ensure_float_list(x)) != num_labels)
        if bad.any():
            idx = bad[bad].index.tolist()[:5]
            raise ValueError(f"{name}: label_vec length mismatch for {bad.sum()} rows. Examples: {idx}")

    # Build HF datasets
    val_text_ds  = to_hf_text(val_df)
    test_text_ds = to_hf_text(test_df)
    val_audio_ds = to_hf_audio(val_df)
    test_audio_ds= to_hf_audio(test_df)

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(args.text_dir)
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_dir).to(device)
    text_model.eval()

    audio_model = AudioClassifier(input_dim=len(AUDIO_COLS), num_labels=num_labels).to(device)
    audio_model.load_state_dict(torch.load(args.audio_pt, map_location=device))
    audio_model.eval()

    # Run models on val for tuning
    val_text_logits,  val_labels = run_text_model(text_model, tokenizer, val_text_ds, args.batch_size)
    val_audio_logits, _          = run_audio_model(audio_model, val_audio_ds, args.batch_size)
    val_text_probs  = sigmoid(val_text_logits)
    val_audio_probs = sigmoid(val_audio_logits)

    # Thresholds: load or tune
    th_store = safe_load_thresholds(args.thresholds_json)
    text_thresholds  = None
    audio_thresholds = None
    if isinstance(th_store, dict):
        if isinstance(th_store.get("text_thresholds"), list) and len(th_store["text_thresholds"]) == num_labels:
            text_thresholds = np.array(th_store["text_thresholds"], dtype=np.float32)
        if isinstance(th_store.get("audio_thresholds"), list) and len(th_store["audio_thresholds"]) == num_labels:
            audio_thresholds = np.array(th_store["audio_thresholds"], dtype=np.float32)

    if text_thresholds is None:
        text_thresholds = tune_thresholds(val_text_probs, val_labels)
    if audio_thresholds is None:
        audio_thresholds = tune_thresholds(val_audio_probs, val_labels)

    # Fusion alpha tuning on val
    alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    best_alpha, best_macro = 0.5, -1.0
    fusion_thresholds = None
    for a in alphas:
        fusion_probs = a * val_text_probs + (1.0 - a) * val_audio_probs
        th = tune_thresholds(fusion_probs, val_labels)
        preds = apply_thresholds(fusion_probs, th)
        macro = f1_score(val_labels, preds, average="macro", zero_division=0)
        if macro > best_macro:
            best_macro = macro
            best_alpha = a
            fusion_thresholds = th

    print(f"[VAL] best alpha={best_alpha:.2f} macro_f1={best_macro:.4f}")

    # --- Evaluate on TEST ---
    test_text_logits,  test_labels = run_text_model(text_model, tokenizer, test_text_ds, args.batch_size)
    test_audio_logits, _           = run_audio_model(audio_model, test_audio_ds, args.batch_size)
    test_text_probs  = sigmoid(test_text_logits)
    test_audio_probs = sigmoid(test_audio_logits)

    # Text
    text_preds = apply_thresholds(test_text_probs, text_thresholds)
    text_agg, text_df = metrics_report(test_labels, text_preds, id2label)
    print("[TEST][TEXT] ", text_agg)

    # Audio
    audio_preds = apply_thresholds(test_audio_probs, audio_thresholds)
    audio_agg, audio_df = metrics_report(test_labels, audio_preds, id2label)
    print("[TEST][AUDIO]", audio_agg)

    # Fusion
    fusion_probs = best_alpha * test_text_probs + (1.0 - best_alpha) * test_audio_probs
    # (Re-use thresholds tuned on val for fusion)
    fusion_preds = apply_thresholds(fusion_probs, fusion_thresholds)
    fusion_agg, fusion_df = metrics_report(test_labels, fusion_preds, id2label)
    print("[TEST][FUSION]", fusion_agg)

    # Save reports
    text_df.to_csv("eval_reports/per_label_text.csv", index=False)
    audio_df.to_csv("eval_reports/per_label_audio.csv", index=False)
    fusion_df.to_csv("eval_reports/per_label_fusion.csv", index=False)

    with open("eval_reports/summary.json", "w") as f:
        json.dump({
            "text": text_agg,
            "audio": audio_agg,
            "fusion": fusion_agg,
            "alpha": best_alpha,
        }, f, indent=2)

    # Save predictions (for error analysis)
    out_pred = pd.DataFrame({
        "y_true": list(test_labels.astype(int)),
        "text_probs": list(test_text_probs.astype(float)),
        "audio_probs": list(test_audio_probs.astype(float)),
        "fusion_probs": list(fusion_probs.astype(float)),
        "text_pred": list(text_preds.astype(int)),
        "audio_pred": list(audio_preds.astype(int)),
        "fusion_pred": list(fusion_preds.astype(int)),
    })
    out_pred.to_parquet("eval_predictions/test_predictions.parquet", index=False)

    print("Saved:")
    print(" - eval_reports/per_label_{text,audio,fusion}.csv")
    print(" - eval_reports/summary.json")
    print(" - eval_predictions/test_predictions.parquet")

if __name__ == "__main__":
    main()
