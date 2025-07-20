import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ------------------------------
# Step 1: Load and preprocess data
# ------------------------------
df = pd.read_csv("counselchat_labels_improved.csv")

# Filter valid label rows
df = df[df["mapped_labels"].notnull()]
df = df[df["mapped_labels"].apply(lambda x: len(eval(x)) > 0)]

# Rename column and convert text
df.rename(columns={"questionText": "text"}, inplace=True)
df["text"] = df["text"].astype(str)

# Remove very short texts
df = df[df["text"].str.len() > 20]

# Extract all unique labels
all_labels = sorted({label for sublist in df["mapped_labels"].apply(eval) for label in sublist})
label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(label2id)

# Multi-hot encode labels
def multi_hot(label_list):
    vec = [0] * num_labels
    for label in eval(label_list):
        if label in label2id:
            vec[label2id[label]] = 1
    return vec

df["label_vector"] = df["mapped_labels"].apply(multi_hot)

# alculate class weights
print("Calculating class weights...")
label_matrix = np.array(df["label_vector"].tolist())
pos_counts = label_matrix.sum(axis=0)
neg_counts = len(df) - pos_counts

class_weights = []
for i in range(num_labels):
    pos = pos_counts[i]
    neg = neg_counts[i]
    
    # More aggressive weighting for rare classes
    if pos < 30:  # Very rare class
        weight = np.sqrt(neg / (pos + 1)) * 2.0
    elif pos < 100:  # Rare class
        weight = np.sqrt(neg / (pos + 1)) * 1.5
    else:  # Normal class
        weight = np.sqrt(neg / (pos + 1))
    
    class_weights.append(weight)
    print(f"{id2label[i]}: {pos} samples, weight={weight:.2f}")

class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ------------------------------
# Step 2: Train/Val/Test Split
# ------------------------------
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# ------------------------------
# Step 3: Create Hugging Face Datasets
# ------------------------------
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["text", "label_vector"]]),
    "validation": Dataset.from_pandas(val_df[["text", "label_vector"]]),
    "test": Dataset.from_pandas(test_df[["text", "label_vector"]])
})

# ------------------------------
# Step 4: Tokenization
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def preprocess(example):
    encodings = tokenizer(
        example["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )
    encodings["labels"] = torch.tensor(example["label_vector"], dtype=torch.float32)
    return encodings

tokenized_dataset = dataset.map(preprocess)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------
# Step 5: Model & Training Setup
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# ------------------------------
# Step 6: Threshold Tuning & Metrics
# ------------------------------
# Dynamic threshold tuning
best_thresholds = None
best_val_f1 = 0

def tune_thresholds_advanced(probs, labels):
    """Better threshold tuning using precision-recall curves"""
    thresholds = []
    
    for i in range(probs.shape[1]):
        if labels[:, i].sum() < 5:  # Very few positive samples
            thresholds.append(0.5)
            continue
        
        # Get precision-recall curve
        precisions, recalls, threshs = precision_recall_curve(labels[:, i], probs[:, i])
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find threshold with best F1
        best_idx = np.argmax(f1_scores)
        best_threshold = threshs[best_idx] if best_idx < len(threshs) else 0.5
        
        # Ensure threshold is not too extreme
        best_threshold = np.clip(best_threshold, 0.1, 0.9)
        
        thresholds.append(best_threshold)
    
    return np.array(thresholds)

def compute_metrics(pred):
    global best_thresholds, best_val_f1
    
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(pred.label_ids)
    probs = torch.sigmoid(logits).numpy()
    labels = labels.numpy()
    
    # Tune thresholds EVERY TIME
    current_thresholds = tune_thresholds_advanced(probs, labels)
    
    # Apply thresholds
    preds = (probs >= current_thresholds).astype(int)
    
    # Calculate metrics
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    # Update best thresholds if better
    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        best_thresholds = current_thresholds
        print(f"\n🔧 New best thresholds found! Macro F1: {macro_f1:.4f}")
    
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

# Custom trainer with weighted loss
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted loss for imbalanced multi-label classification
        **kwargs handles any extra arguments from newer transformers versions
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted BCE loss for class imbalance
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ------------------------------
# Step 7: Trainer
# ------------------------------
training_args = TrainingArguments(
    output_dir="./counselchat_weighted_model", 
    evaluation_strategy="steps", 
    eval_steps=50, 
    save_strategy="steps",
    save_steps=100,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    label_smoothing_factor=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=25,
    fp16=torch.cuda.is_available(),
)

# Use WeightedLossTrainer instead of Trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ------------------------------
# Step 8: Train and Save
# ------------------------------
print("\nStarting WEIGHTED ClinicalBERT fine-tuning...")
print("This should perform MUCH better than before!")
print("="*50)

trainer.train()
print("Training complete!")

# Save model and tokenizer
trainer.save_model("./counselchat_weighted_model/best_model")
tokenizer.save_pretrained("./counselchat_weighted_model/best_model")

# Save thresholds
import json
threshold_info = {
    "thresholds": best_thresholds.tolist() if best_thresholds is not None else None,
    "best_val_f1": best_val_f1,
    "class_weights": class_weights.tolist()
}
with open("./counselchat_weighted_model/thresholds.json", "w") as f:
    json.dump(threshold_info, f)

print(f"\nModel saved with best validation macro F1: {best_val_f1:.4f}")
print("Saved to: ./counselchat_weighted_model/")