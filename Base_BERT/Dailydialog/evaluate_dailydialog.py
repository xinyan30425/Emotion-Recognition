import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import DataCollatorWithPadding

# Load DailyDialog with custom code permission
dataset = load_dataset("daily_dialog", trust_remote_code=True)
test_raw = dataset["test"]

# for i in range(10):
#     print(f"Dialog {i}:")
#     print("Utterances:", test_raw[i]["dialog"])
#     print("Emotions:  ", test_raw[i]["emotion"])
#     print("=" * 50)
    

# Define emotion labels
emotion_labels = [
    "no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"
]
num_labels = len(emotion_labels)
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./dailydialog_finetuned/fine-tuned-bert")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flatten test dialogue into utterances
def flatten(dialogues):
    flat = []
    for dialog, emotions in zip(dialogues["dialog"], dialogues["emotion"]):
        for utt, emo in zip(dialog, emotions):
            flat.append({"text": utt, "label": emo})
    return flat

from datasets import Dataset
test_dataset = Dataset.from_list(flatten(test_raw))

# Tokenization
def preprocess(example):
    encodings = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    encodings["label"] = example["label"]
    return encodings

tokenized_test = test_dataset.map(preprocess)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=emotion_labels, zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }

# Run evaluation using Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print("\nEvaluating fine-tuned BERT on DailyDialog test set...")
results = trainer.evaluate(eval_dataset=tokenized_test)
print("\nFinal Evaluation Results:")
print(results)
