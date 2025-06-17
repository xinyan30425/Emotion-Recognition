# BERT Emotion Classifier – Model Files Overview

This folder contains the necessary files to load and run a locally stored BERT model for emotion classification. Below is an explanation of each file.

downloaded from:

https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main
---

## File Descriptions

### `pytorch_model.bin`
- The **pre-trained model weights**.
- It is a binary file containing the pre-trained model's weights (i.e., all the learned parameters from BERT).
- (e.g., attention layers, embeddings, classifier head).
- This is the core of the model — the "brain" that learned from data.

---

### `config.json`
- Defines the **model architecture and hyperparameters**.
- Example fields include:
  - `hidden_size`: 768
  - `num_attention_heads`: 12
  - `num_hidden_layers`: 12
  - `num_labels`: Number of emotion classes
- Required to initialize the model structure before loading weights.

---

### `vocab.txt`
- The **vocabulary list** used by the tokenizer.
- Contains all tokens/subwords known to the model (one token per line).
- Used during tokenization to convert text into token IDs.
- BERT uses WordPiece, so rare words are split into subword units.

---

### `tokenizer_config.json`
- Configuration for the tokenizer behavior.
- Defines properties like:
  - Whether to lowercase text
  - The tokenizer class used
  - Max input length
  - Padding/truncation behavior (if defined)

---

### `special_tokens_map.json`
- Maps **special roles** to their corresponding tokens.
- Example mappings:
  - `"cls_token": "[CLS]"`
  - `"sep_token": "[SEP]"`
  - `"pad_token": "[PAD]"`
  - `"unk_token": "[UNK]"`
  - `"mask_token": "[MASK]"`

---

## How to Load This Model in Code

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load from current directory
tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForSequenceClassification.from_pretrained(".")
