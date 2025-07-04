import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score

# Load DailyDialog
dataset = load_dataset("daily_dialog", trust_remote_code=True)

# Define emotion labels (based on DailyDialog docs)
emotion_labels = [
    "no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"
]
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(emotion_labels)

# Flatten dialogues into (utterance, label) pairs
def extract_utterances(example):
    return {
        "text": example["dialog"],
        "emotion": example["emotion"]
    }

# Flatten each dialogue
def flatten(dialogues):
    flat = []
    for dialog, emotions in zip(dialogues["dialog"], dialogues["emotion"]):
        for utt, emo in zip(dialog, emotions):
            flat.append({"text": utt, "label": emo})
    return flat

train = flatten(dataset["train"])
val = flatten(dataset["validation"])
test = flatten(dataset["test"])

from datasets import Dataset, DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_list(train),
    "validation": Dataset.from_list(val),
    "test": Dataset.from_list(test),
})

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = batch["label"]
    return encoding

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

# Training setup
training_args = TrainingArguments(
    output_dir="./dailydialog_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./dailydialog_logs",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train
print("Starting fine-tuning on DailyDialog...")
trainer.train()
print("Training complete!")

# Save model
print("Saving fine-tuned model...")
trainer.save_model("./dailydialog_finetuned/fine-tuned-bert")
tokenizer.save_pretrained("./dailydialog_finetuned/fine-tuned-bert")
print("Model saved successfully.")
