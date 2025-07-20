import pandas as pd

# Load the dataset
df = pd.read_csv("counselchat-data.csv")

# Get unique topics
unique_topics = df["topics"].unique()
    
label_keywords = {
    "anxiety": "Anxiety",
    "depression": "Depression",
    "anger": "Anger",
    "trauma": "Trauma",
    "stress": "Stress",
    "grief": "Grief",
    "loss": "Grief",
    "relationship": "Relationships",
    "marriage": "Relationships",
    "divorce": "Relationships",
    "dissolution": "Relationships",
    "parenting": "Parenting",
    "children": "Parenting",
    "adolescents": "Parenting",
    "family": "Family Conflict",
    "self-esteem": "Self-esteem",
    "intimacy": "Intimacy",
    "sexuality": "Human Sexuality",
    "lgbtq": "LGBTQ",
    "substance": "Substance Abuse",
    "addiction": "Substance Abuse",
    "eating": "Eating Disorders",
    "workplace": "Workplace",
    "career": "Workplace",
    "spirituality": "Spirituality",
    "ethics": "Professional Ethics",
    "legal": "Professional Ethics",
    "diagnosis": "Diagnosis",
    "sleep": "Sleep",
    "behavioral": "Behavioral Change",
    "military": "Military",
    "self-harm": "Self-harm",
    "counseling": "Counseling"
}

def map_topic_to_labels(topic_str):
    if not isinstance(topic_str, str):
        return ["Other"]  # Handle NaN or non-string values

    topic_str = topic_str.lower()
    matched_labels = set()

    for keyword, label in label_keywords.items():
        if keyword in topic_str:
            matched_labels.add(label)

    if not matched_labels:
        matched_labels.add("Other")  # fallback
    return list(matched_labels)


# Apply the mapping
df["mapped_labels"] = df["topics"].apply(map_topic_to_labels)


from collections import Counter

# If mapped_labels is a list of labels per row (multilabel), explode it first
all_labels = df["mapped_labels"].explode()

# Count occurrences
label_counts = Counter(all_labels)

# Filter labels that appear only once
rare_labels = {label: count for label, count in label_counts.items() if count == 2}

# Print count and the labels
print(f"\n Number of labels with only 1 sample: {len(rare_labels)}")
print("These rare labels are:")
for label in rare_labels:
    print(f"- {label}")