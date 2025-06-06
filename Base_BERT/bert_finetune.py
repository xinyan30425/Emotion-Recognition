import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Define emotion label mapping
emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')  # or 'macro' for balanced F1
    return {"accuracy": accuracy, "f1": f1}

# Load MELD CSV files
dataset = load_dataset('csv', 
    data_files={
        "train": "MELDRaw/train/train_sent_emo.csv",
        "validation": "MELDRaw/dev_sent_emo.csv",
        "test": "MELDRaw/test_sent_emo.csv"
    }
)

# Keep only Utterance + Emotion columns, and convert Emotion to int
def preprocess(row):
    return {
        "text": row["Utterance"],
        "label": label2id[row["Emotion"]]
    }

dataset = dataset.map(preprocess)
dataset = dataset.remove_columns([
    'Sr No.', 'Utterance_ID', 'Speaker', 'Sentiment', 'Dialogue_ID',
    'Season', 'Episode', 'StartTime', 'EndTime', 'Emotion'
])

# Tokenize utterances
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(emotion_labels),
    id2label=id2label,
    label2id=label2id
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)

# Train using Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine-tuned-meld-emotion")