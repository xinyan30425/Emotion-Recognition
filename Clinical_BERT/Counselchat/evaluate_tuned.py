import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, classification_report

# ----- Load Data -----
df = pd.read_csv("counselchat_labels_improved.csv")
df = df[df["mapped_labels"].notnull()]
df = df[df["mapped_labels"].apply(lambda x: len(eval(x)) > 0)]
df.rename(columns={"questionText": "text"}, inplace=True)
df["text"] = df["text"].astype(str)

# ----- Label Mapping -----
all_labels = sorted({label for row in df["mapped_labels"].apply(eval) for label in row})
label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(label2id)

# ----- Multi-hot Encode -----
def encode_labels(row):
    vec = [0] * num_labels
    for label in eval(row["mapped_labels"]):
        if label in label2id:
            vec[label2id[label]] = 1
    return vec

df["label_vector"] = df.apply(encode_labels, axis=1)

# ----- Create HuggingFace Dataset -----
dataset = Dataset.from_pandas(df[["text", "label_vector"]])

# ----- Tokenization -----
tokenizer = AutoTokenizer.from_pretrained("./counselchat_weighted_model/best_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "./counselchat_weighted_model/best_model",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(example):
    enc = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    enc["labels"] = torch.tensor(example["label_vector"], dtype=torch.float32)
    return enc

tokenized = dataset.map(preprocess)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ----- Threshold Tuning -----
cached_thresholds = None

def tune_thresholds(probs, labels, steps=np.arange(0.1, 0.9, 0.05)):
    thresholds = []
    for i in range(probs.shape[1]):
        best_f1, best_th = 0, 0.5
        for th in steps:
            preds = (probs[:, i] >= th).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        thresholds.append(best_th)
    return np.array(thresholds)

def compute_metrics(pred):
    global cached_thresholds
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(pred.label_ids)
    probs = torch.sigmoid(logits).numpy()
    labels = labels.numpy()

    if cached_thresholds is None:
        cached_thresholds = tune_thresholds(probs, labels)
        print("🔧 Tuned thresholds:", cached_thresholds)

    preds = (probs >= cached_thresholds).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, target_names=[id2label[i] for i in range(num_labels)], zero_division=0))

    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)
    }

# ----- Run Evaluation -----
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

print("Evaluating model on CounselChat data (multi-label)...")
results = trainer.evaluate(eval_dataset=tokenized)
print("\nFinal Evaluation Results:")
print(results)
