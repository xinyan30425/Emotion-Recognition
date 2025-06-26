import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score

# Load GoEmotions simplified dataset
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# Define label info
emotion_labels = dataset["train"].features["labels"].feature.names
label2id = {label: i for i, label in enumerate(emotion_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(emotion_labels)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    encodings = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    
    # Create float32 label matrix
    label_matrix = np.zeros((len(batch["labels"]), num_labels), dtype=np.float32)
    for i, label_list in enumerate(batch["labels"]):
        label_matrix[i, label_list] = 1.0
    
    encodings["labels"] = label_matrix.tolist()  # Convert to list to avoid dtype issues
    return encodings

# Tokenize the datasets
tokenized_dataset = dataset.map(preprocess, batched=True)

# Remove the original columns to avoid conflicts
tokenized_dataset = tokenized_dataset.remove_columns(["text", "id"])

# Custom data collator to ensure labels are float tensors
from transformers import DataCollatorWithPadding
import torch

class MultiLabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # Ensure labels are float tensors
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

# Initialize data collator
data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

# Metrics for multi-label
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    # Calculate per-label accuracy for more insights
    per_label_acc = ((preds == labels).sum(axis=0) / len(labels)).tolist()
    
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "mean_per_label_acc": np.mean(per_label_acc)
    }

# Model for multi-label
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./goemotions_finetuned",
    evaluation_strategy="epoch",  # Updated parameter name
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./goemotions_logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision if GPU available
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,  # Use custom data collator
    compute_metrics=compute_metrics,
)

# Train
print("Starting training...")
trainer.train()

# Save the model
trainer.save_model("./goemotions_finetuned/fine-tuned-bert")
