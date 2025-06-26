import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import f1_score, classification_report
from transformers import DataCollatorWithPadding

# Load GoEmotions simplified dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
test_dataset = dataset["test"]

# Emotion label info
emotion_labels = dataset["train"].features["labels"].feature.names
num_labels = len(emotion_labels)
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}

# Load tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./goemotions_finetuned/fine-tuned-bert")
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing function (same as training)
def preprocess(batch):
    encodings = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    label_matrix = np.zeros((len(batch["labels"]), num_labels), dtype=np.float32)
    for i, label_list in enumerate(batch["labels"]):
        label_matrix[i, label_list] = 1.0
    encodings["labels"] = label_matrix.tolist()
    return encodings

# Tokenize test set
tokenized_test = test_dataset.map(preprocess, batched=True)
tokenized_test = tokenized_test.remove_columns(["text", "id"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Custom data collator
class MultiLabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        
        # Ensure each label is a float list of length `num_labels`
        labels = [torch.tensor(label, dtype=torch.float32) for label in labels]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Stack into a batch tensor: shape = (batch_size, num_labels)
        batch["labels"] = torch.stack(labels)
        return batch

data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

# Evaluation metrics
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    sample_accuracy = (preds == labels).all(axis=1).mean()  # subset accuracy

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "subset_accuracy": sample_accuracy,
    }

# Run evaluation using Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print("\nEvaluating fine-tuned model on GoEmotions test set...")
results = trainer.evaluate(eval_dataset=tokenized_test)
print("\nFinal Evaluation Results:")
print(results)

