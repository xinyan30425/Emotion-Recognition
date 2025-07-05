import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Define emotion labels
emotion_labels = [
    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive",
    "confident", "content", "devastated", "disappointed", "disgusted",
    "embarrassed", "excited", "faithful", "grateful", "guilty", "hopeful",
    "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared",
    "proud", "sad", "sentimental", "surprised", "trusting", "terrified"
]
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

# Clean and load CSVs into Hugging Face datasets
def clean_and_load(csv_path):
    df = pd.read_csv(csv_path,quotechar= '"',on_bad_lines='skip')
    df = df[df['context'].isin(emotion_labels)].copy()
    df = df.rename(columns={"utterance": "text", "context": "label_text"})
    df["label"] = df["label_text"].map(label2id)
    df = df[["text", "label"]].dropna()
    return Dataset.from_pandas(df)

dataset = {
    "train": clean_and_load("empatheticdialogues/train.csv"),
    "validation": clean_and_load("empatheticdialogues/valid.csv"),
    "test": clean_and_load("empatheticdialogues/test.csv")
}

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = {split: dataset[split].map(tokenize, batched=True) for split in dataset}
for split in tokenized_dataset:
    tokenized_dataset[split].set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(emotion_labels),
    id2label=id2label,
    label2id=label2id
)

# Training args
training_args = TrainingArguments(
    output_dir="./empathetic_dialogues_results",
    logging_dir="./empathetic_dialogues_logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Run training
trainer.train()
trainer.save_model("./empathetic_dialogues_finetuned_model/fine-tuned-bert")

