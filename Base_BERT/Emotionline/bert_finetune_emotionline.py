import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# Load EmotionLines dataset
dataset = load_dataset("emotion")  # already split into train/validation/test

# Get labels
emotion_labels = dataset["train"].features["label"].names
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(emotion_labels)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess function
def preprocess(batch):
    encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = batch["label"]
    return encoding

# Apply tokenization
tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro")
    }

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./emotionlines_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./emotionlines_logs",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train
print("Fine-tuning BERT on EmotionLines dataset...")
trainer.train()

# Save model and tokenizer
trainer.save_model("./emotionlines_finetuned/fine-tuned-bert")
tokenizer.save_pretrained("./emotionlines_finetuned/fine-tuned-bert")
