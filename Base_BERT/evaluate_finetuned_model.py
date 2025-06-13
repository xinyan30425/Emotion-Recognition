import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Load label mappings
emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

# Load fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-meld-emotion")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess test dataset
dataset = load_dataset('csv', data_files={"test": "MELDRaw/test_sent_emo.csv"})

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

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# Run evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(tokenized_dataset["test"])
print("Evaluation Results on Test Set:")
print(results)

