# train_early_fusion.py
import json, torch, torch.nn as nn, pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
import numpy as np

MODEL_PATH = "../counselchat/counselchat_weighted_model/best_model"  # or your DAIC-WOZ fine-tuned path
AUDIO_COLS = ["pitch_mean","energy"] + [f"mfcc_{i}" for i in range(13)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load data & labels -----
train_df = pd.read_parquet("mm_train.parquet")
val_df   = pd.read_parquet("mm_val.parquet")
test_df  = pd.read_parquet("mm_test.parquet")
label_meta = json.load(open("mm_labels.json"))
label2id = label_meta["label2id"]; id2label = {int(k):v for k,v in label_meta["id2label"].items()}
num_labels = len(label2id)

# ----- Tokenizer -----
from transformers import AutoConfig
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def to_hf(df):
    return Dataset.from_pandas(pd.DataFrame({
        "text": df["text"].tolist(),
        "audio": df[AUDIO_COLS].values.tolist(),
        "labels": df["label_vec"].tolist()
    }))

dataset = DatasetDict({
    "train": to_hf(train_df),
    "validation": to_hf(val_df),
    "test": to_hf(test_df),
})

def preprocess(ex):
    tok = tokenizer(ex["text"], padding="max_length", truncation=True, max_length=256)
    tok["audio"] = ex["audio"]
    tok["labels"] = ex["labels"]
    return tok

dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

# set format for torch (keep audio as float tensor)
columns = ["input_ids","attention_mask","audio","labels"]
dataset.set_format(type="torch", columns=columns)

# ----- Model -----
class EarlyFusionModel(nn.Module):
    def __init__(self, bert_name_or_path, audio_dim, num_labels, freeze_bert=False):
        super().__init__()
        self.text_enc = AutoModel.from_pretrained(bert_name_or_path)
        hidden = self.text_enc.config.hidden_size
        if freeze_bert:
            for p in self.text_enc.parameters(): p.requires_grad = False

        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden + hidden//2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask, audio, labels=None):
        # text encoding
        out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        # pooled representation (use CLS)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            text_vec = out.pooler_output
        else:
            text_vec = out.last_hidden_state[:,0,:]  # CLS token

        audio = audio.float()
        audio_vec = self.audio_proj(audio)
        fused = torch.cat([text_vec, audio_vec], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

model = EarlyFusionModel(MODEL_PATH, audio_dim=len(AUDIO_COLS), num_labels=num_labels, freeze_bert=False)

# ----- Metrics with per-class threshold tuning -----
def tune_thresholds(probs, labels):
    # simple but strong: per-class grid search
    thresholds = []
    for i in range(probs.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1,0.9,17):
            pred = (probs[:,i] >= t).astype(int)
            f1 = f1_score(labels[:,i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return np.array(thresholds)

best_thresholds = None
best_macro = 0.0

def compute_metrics(eval_pred):
    global best_thresholds, best_macro
    logits, labels = eval_pred
    probs = 1/(1+np.exp(-logits))
    thresholds = tune_thresholds(probs, labels)
    preds = (probs >= thresholds).astype(int)
    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    if macro > best_macro:
        best_macro = macro
        best_thresholds = thresholds
    return {"micro_f1": micro, "macro_f1": macro}

# ----- Trainer -----
args = TrainingArguments(
    output_dir="mm_early_fusion",
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=25,
    eval_steps=100,
    save_steps=100,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
)

class PassThroughTrainer(Trainer):
    pass

trainer = PassThroughTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model("mm_early_fusion/best_model")
tokenizer.save_pretrained("mm_early_fusion/best_model")
if best_thresholds is not None:
    import json
    json.dump({"thresholds": best_thresholds.tolist()}, open("mm_early_fusion/thresholds.json","w"))
