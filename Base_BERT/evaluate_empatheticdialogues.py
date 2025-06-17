import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Emotion label set used in EmpatheticDialogues
emotion_labels = [
    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive",
    "confident", "content", "devastated", "disappointed", "disgusted",
    "embarrassed", "excited", "faithful", "grateful", "guilty", "hopeful",
    "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared",
    "proud", "sad", "sentimental", "surprised", "trusting", "terrified"
]
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

# Load fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./empathetic_dialogues_finetuned_model/fine-tuned-bert")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Clean and load test dataset
def clean_and_load(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")  # Skip malformed rows
    df = df[df['context'].isin(emotion_labels)].copy()
    df = df.rename(columns={"utterance": "text", "context": "label_text"})
    df["label"] = df["label_text"].map(label2id)
    df = df[["text", "label"]].dropna()
    return Dataset.from_pandas(df)

# Load cleaned dataset
dataset = {"test": clean_and_load("empatheticdialogues/test.csv")}

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset["test"].map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Run evaluation
results = trainer.evaluate(tokenized_dataset)
print("Evaluation Results on EmpatheticDialogues Test Set:")
print(results)

