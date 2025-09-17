# train_late_fusion.py
# Late-fusion training for multi-label text (ClinicalBERT) + audio (MLP)
# - Robust label dtype/shape handling
# - Custom data collators (float32 labels)
# - Defensive preprocess + assertions
# - Stable compute_metrics + threshold tuning

import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import EvalPrediction


# -----------------------
# 1) Config
# -----------------------
MODEL_PATH = "../counselchat/counselchat_weighted_model/best_model"
AUDIO_COLS = ["pitch_mean", "energy"] + [f"mfcc_{i}" for i in range(13)]
MAX_LEN = 256
TEXT_OUTDIR = "mm_late_fusion/text"
AUDIO_OUTDIR = "mm_late_fusion/audio"
THRESH_PATH = "mm_late_fusion/thresholds.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# 2) Load data
# -----------------------
train_df = pd.read_parquet("mm_train.parquet")
val_df   = pd.read_parquet("mm_val.parquet")
test_df  = pd.read_parquet("mm_test.parquet")

label_meta = json.load(open("mm_labels.json"))
label2id   = label_meta["label2id"]
# ensure keys are int for id2label
id2label   = {int(k): v for k, v in label_meta["id2label"].items()}
num_labels = len(label2id)


# -----------------------
# 3) Basic sanitation on dataframes
# -----------------------
def _ensure_list_floats(x):
    # Accepts list-like (of numbers/strings) or np array; returns list[float]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in list(x)]
    # Fallback: singletons -> wrap
    return [float(x)]

for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    # Ensure label_vec exists and has correct length
    assert "label_vec" in df.columns, f"{split_name} missing 'label_vec' column"
    df["label_vec"] = df["label_vec"].apply(_ensure_list_floats)
    bad_lens = df["label_vec"].apply(len) != num_labels
    if bad_lens.any():
        bad_idx = bad_lens[bad_lens].index.tolist()[:5]
        raise ValueError(
            f"{split_name}: Found {bad_lens.sum()} rows with label_vec length != num_labels={num_labels}. "
            f"Examples idx: {bad_idx}"
        )
    # Ensure audio columns exist
    for c in AUDIO_COLS:
        if c not in df.columns:
            raise ValueError(f"{split_name}: missing required audio feature column '{c}'")

    # Ensure text exists
    if "text" not in df.columns:
        raise ValueError(f"{split_name}: missing required 'text' column")


# -----------------------
# 4) Tokenizer + datasets
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def to_hf_text(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({
        "text": df["text"].tolist(),
        "labels": df["label_vec"].tolist(),
    }))

def to_hf_audio(df: pd.DataFrame) -> Dataset:
    # pack selected columns into a feature vector
    return Dataset.from_pandas(pd.DataFrame({
        "audio": df[AUDIO_COLS].values.tolist(),  # list[float] of len=len(AUDIO_COLS)
        "labels": df["label_vec"].tolist(),
    }))

dataset_text = DatasetDict({
    "train": to_hf_text(train_df),
    "validation": to_hf_text(val_df),
    "test": to_hf_text(test_df),
})

dataset_audio = DatasetDict({
    "train": to_hf_audio(train_df),
    "validation": to_hf_audio(val_df),
    "test": to_hf_audio(test_df),
})


def preprocess_text(ex):
    # Tokenize
    tok = tokenizer(
        ex["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_token_type_ids=True
    )
    # Keep labels as python lists of floats (collator will cast to float32 tensors)
    # Also validate shape
    labs = []
    for lab in ex["labels"]:
        lab = [float(v) for v in lab]
        if len(lab) != num_labels:
            raise ValueError(f"Label length {len(lab)} != num_labels {num_labels}")
        labs.append(lab)
    tok["labels"] = labs
    return tok

# Map -> remove original "text"
dataset_text = dataset_text.map(preprocess_text, batched=True, remove_columns=["text"])


# -----------------------
# 5) Models
# -----------------------
# Text model
text_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True,  # head will re-init if num_labels differs
).to(device)

# Persist label maps in config for downstream use
text_model.config.label2id = label2id
text_model.config.id2label = id2label
text_model.config.problem_type = "multi_label_classification"

# Audio model (simple MLP)
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

    def forward(self, audio=None, labels=None):
        x = audio.float()
        logits = self.net(x)
        loss = None
        if labels is not None:
            labels = labels.float()
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

audio_model = AudioClassifier(input_dim=len(AUDIO_COLS), num_labels=num_labels).to(device)


# -----------------------
# 6) Data collators (dtype-safe)
# -----------------------
def collate_text(batch: List[Dict]):
    # Expect keys from preprocess_text: input_ids, attention_mask, token_type_ids (maybe), labels
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)

    out = {"input_ids": input_ids, "attention_mask": attention_mask}

    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.tensor([ex["token_type_ids"] for ex in batch], dtype=torch.long)

    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.float32)
    out["labels"] = labels
    return out

def collate_audio(batch: List[Dict]):
    audio = torch.tensor([ex["audio"] for ex in batch], dtype=torch.float32)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.float32)
    return {"audio": audio, "labels": labels}


# -----------------------
# 7) Metrics + threshold tuning
# -----------------------
def tune_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Per-class threshold sweep in [0.1, 0.9] to maximize F1.
    probs: (N, C) sigmoid outputs
    labels: (N, C) 0/1
    """
    thresholds = []
    for i in range(probs.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 17):
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return np.array(thresholds, dtype=np.float32)

best_thresholds_text, best_macro_text = None, 0.0
best_thresholds_audio, best_macro_audio = None, 0.0

def compute_metrics(eval_pred: EvalPrediction, tag: str = "text"):
    global best_thresholds_text, best_macro_text
    global best_thresholds_audio, best_macro_audio

    logits = eval_pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = eval_pred.label_ids

    probs = 1.0 / (1.0 + np.exp(-logits))
    thresholds = tune_thresholds(probs, labels)
    preds = (probs >= thresholds).astype(int)

    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)

    if tag == "text" and macro > best_macro_text:
        best_macro_text, best_thresholds_text = float(macro), thresholds
    if tag == "audio" and macro > best_macro_audio:
        best_macro_audio, best_thresholds_audio = float(macro), thresholds

    return {"micro_f1": float(micro), "macro_f1": float(macro)}


# -----------------------
# 8) TrainingArguments
# -----------------------
args_text = TrainingArguments(
    output_dir=TEXT_OUTDIR,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=False,  # keep False to avoid fp16+BCE dtype pitfalls; re-enable later if desired
    save_total_limit=2,
    logging_steps=50,
)

args_audio = TrainingArguments(
    output_dir=AUDIO_OUTDIR,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=False,  # simple MLP; keep in fp32 for stability
    save_total_limit=2,
    logging_steps=50,
)


# -----------------------
# 9) Trainers
# -----------------------
trainer_text = Trainer(
    model=text_model,
    args=args_text,
    train_dataset=dataset_text["train"],
    eval_dataset=dataset_text["validation"],
    tokenizer=tokenizer,
    data_collator=collate_text,
    compute_metrics=lambda pred: compute_metrics(pred, tag="text"),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer_audio = Trainer(
    model=audio_model,
    args=args_audio,
    train_dataset=dataset_audio["train"],
    eval_dataset=dataset_audio["validation"],
    data_collator=collate_audio,
    compute_metrics=lambda pred: compute_metrics(pred, tag="audio"),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)


# -----------------------
# 10) Train + Save
# -----------------------
def main():
    # Final preflight asserts on one sample after mapping
    ex_text = dataset_text["train"][0]
    assert "labels" in ex_text and len(ex_text["labels"]) == num_labels, \
        f"text sample labels malformed: {ex_text.get('labels')}"
    ex_audio = dataset_audio["train"][0]
    assert "labels" in ex_audio and len(ex_audio["labels"]) == num_labels, \
        f"audio sample labels malformed: {ex_audio.get('labels')}"
    assert len(ex_audio["audio"]) == len(AUDIO_COLS), \
        f"audio feature length {len(ex_audio['audio'])} != {len(AUDIO_COLS)}"

    print("Training text model...")
    trainer_text.train()
    trainer_text.save_model(f"{TEXT_OUTDIR}/best_model")
    tokenizer.save_pretrained(f"{TEXT_OUTDIR}/best_model")

    print("Training audio model...")
    trainer_audio.train()
    torch.save(audio_model.state_dict(), f"{AUDIO_OUTDIR}/best_model.pt")

    # Save learned thresholds (may be None if validation never improved)
    to_save = {
        "text_thresholds": (best_thresholds_text.tolist() if best_thresholds_text is not None else None),
        "audio_thresholds": (best_thresholds_audio.tolist() if best_thresholds_audio is not None else None),
        "label2id": label2id,
        "id2label": id2label,
    }
    # Ensure dir exists
    import os
    os.makedirs("mm_late_fusion", exist_ok=True)
    with open(THRESH_PATH, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved thresholds + label maps to {THRESH_PATH}")

if __name__ == "__main__":
    main()
