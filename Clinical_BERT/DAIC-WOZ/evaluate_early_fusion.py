# evaluate_early_fusion.py
import json, torch, numpy as np, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, Trainer
from sklearn.metrics import f1_score, classification_report

from train_early_fusion import EarlyFusionModel, AUDIO_COLS  # reuse model definition

# --------------------
# 1. Paths & metadata
# --------------------
MODEL_DIR = "mm_early_fusion/best_model"
TEST_PATH = "mm_test.parquet"
LABEL_META = "mm_labels.json"
THRESH_FILE = "mm_early_fusion/thresholds.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_meta = json.load(open(LABEL_META))
label2id = label_meta["label2id"]
id2label = {int(k): v for k, v in label_meta["id2label"].items()}
num_labels = len(label2id)

thresholds = None
if os.path.exists(THRESH_FILE):
    thresholds = np.array(json.load(open(THRESH_FILE))["thresholds"])

# --------------------
# 2. Load data
# --------------------
test_df = pd.read_parquet(TEST_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def to_hf(df):
    return Dataset.from_pandas(pd.DataFrame({
        "text": df["text"].tolist(),
        "audio": df[AUDIO_COLS].values.tolist(),
        "labels": df["label_vec"].tolist()
    }))

test_ds = to_hf(test_df)

def preprocess(ex):
    tok = tokenizer(ex["text"], padding="max_length", truncation=True, max_length=256)
    tok["audio"] = ex["audio"]
    tok["labels"] = ex["labels"]
    return tok

test_ds = test_ds.map(preprocess, batched=True, remove_columns=["text"])
test_ds.set_format(type="torch", columns=["input_ids","attention_mask","audio","labels"])

# --------------------
# 3. Load model
# --------------------
model = EarlyFusionModel(
    bert_name_or_path=MODEL_DIR,
    audio_dim=len(AUDIO_COLS),
    num_labels=num_labels,
    freeze_bert=False
).to(device)

trainer = Trainer(model=model, tokenizer=tokenizer)

# --------------------
# 4. Predict
# --------------------
preds = trainer.predict(test_ds)
logits = preds.predictions
labels = preds.label_ids
probs = 1/(1+np.exp(-logits))

# Use tuned thresholds if available
if thresholds is None:
    thresholds = np.array([0.5] * num_labels)

bin_preds = (probs >= thresholds).astype(int)

# --------------------
# 5. Metrics
# --------------------
micro_f1 = f1_score(labels, bin_preds, average="micro", zero_division=0)
macro_f1 = f1_score(labels, bin_preds, average="macro", zero_division=0)

print("\n=== Early Fusion Evaluation ===")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}\n")

# Per-class results
print("Per-class F1:")
print(classification_report(labels, bin_preds, target_names=[id2label[i] for i in range(num_labels)], zero_division=0))
