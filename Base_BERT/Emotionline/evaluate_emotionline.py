import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load EmotionX Friends evaluation data
with open("friends_eval_gold.json", "r") as f:
    raw_data = json.load(f)

# Flatten utterances
flat_data = []
for dialog in raw_data:
    for turn in dialog:
        if turn["emotion"] in ["sadness", "joy", "love", "anger", "fear", "surprise"]:
            flat_data.append({
                "text": turn["utterance"],
                "label_text": turn["emotion"]
            })

# Define label mapping consistent with EmotionLines
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

# Convert text labels to IDs
for entry in flat_data:
    entry["label"] = label2id[entry["label_text"]]

# Load dataset
dataset = Dataset.from_list(flat_data)

# Load tokenizer and model
model_path = "./emotionlines_finetuned/fine-tuned-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocess
def tokenize(batch):
    encoded = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    encoded["labels"] = batch["label"]
    return encoded

tokenized_dataset = dataset.map(tokenize)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Metrics function
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro")
    }

# Run evaluation using Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Evaluating fine-tuned BERT on EmotionX (Friends)...")
results = trainer.evaluate(eval_dataset=tokenized_dataset)

# Print results
print("\n📊 Final Evaluation Metrics:")
print(results)

# Optional: full classification report
preds = trainer.predict(tokenized_dataset)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)
print("\n🔍 Classification Report:")
print(classification_report(y_true, y_pred, target_names=emotion_labels))
