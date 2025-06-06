from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model from local folder
tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForSequenceClassification.from_pretrained("./")

# Define emotion labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Input text
# List of input texts to test
texts = [
    "I feel very sad and alone.",
    "I'm so happy and grateful today!",
    "You make me feel loved and appreciated.",
    "This makes me so angry!",
    "I'm really scared of what might happen.",
    "Wow, what a surprise!",
    "I miss my family and it hurts.",
    "That dog is adorable and makes me smile.",
    "I can't believe this happened again!"
]

# Predict emotion for each input
model.eval()
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
        print(f"Text: {text}\n→ Predicted emotion: {labels[predicted_class]}\n")