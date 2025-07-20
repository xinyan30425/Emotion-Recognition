import pandas as pd
import re
from collections import Counter

# More sophisticated label mapping with context awareness
def create_improved_label_mapping():
    """Create a more nuanced label mapping system"""
    
    # Define primary keywords and their variations
    label_patterns = {
        "Anxiety": {
            "primary": ["anxiety", "anxious", "panic", "worry", "nervous"],
            "secondary": ["fear", "phobia", "stress", "overwhelm"],
            "exclude": ["depression and anxiety"]  # Will be handled separately
        },
        "Depression": {
            "primary": ["depression", "depressed", "sad", "hopeless", "suicidal"],
            "secondary": ["mood", "despair", "melancholy", "empty"],
            "exclude": []
        },
        "Anger": {
            "primary": ["anger", "angry", "rage", "frustrated", "irritable"],
            "secondary": ["aggressive", "hostile", "resentment"],
            "exclude": []
        },
        "Trauma": {
            "primary": ["trauma", "ptsd", "abuse", "assault", "violence"],
            "secondary": ["flashback", "nightmare", "trigger"],
            "exclude": []
        },
        "Stress": {
            "primary": ["stress", "pressure", "burnout", "overwhelm"],
            "secondary": ["tension", "strain", "exhausted"],
            "exclude": ["post traumatic stress"]  # This is trauma
        },
        "Grief": {
            "primary": ["grief", "loss", "bereave", "mourn", "death"],
            "secondary": ["funeral", "passing", "widow"],
            "exclude": []
        },
        "Relationships": {
            "primary": ["relationship", "marriage", "divorce", "partner", "spouse"],
            "secondary": ["dating", "breakup", "separation", "couple"],
            "exclude": ["family relationship"]  # This goes to family
        },
        "Family Conflict": {
            "primary": ["family conflict", "family problem", "parent child"],
            "secondary": ["sibling", "in-law", "family dynamic"],
            "exclude": []
        },
        "Parenting": {
            "primary": ["parenting", "child", "teen", "adolescent", "kid"],
            "secondary": ["discipline", "school", "behavior problem"],
            "exclude": ["inner child", "childhood trauma"]  # These are different
        },
        "Self-esteem": {
            "primary": ["self esteem", "confidence", "self worth", "insecur"],
            "secondary": ["inadequate", "inferior", "self doubt"],
            "exclude": []
        },
        "Intimacy": {
            "primary": ["intimacy", "sexual problem", "affection", "closeness"],
            "secondary": ["emotional connection", "physical touch"],
            "exclude": []
        },
        "Human Sexuality": {
            "primary": ["sexuality", "sexual orientation", "sexual identity"],
            "secondary": ["heterosexual", "homosexual", "bisexual", "asexual"],
            "exclude": []
        },
        "LGBTQ": {
            "primary": ["lgbtq", "gay", "lesbian", "transgender", "queer"],
            "secondary": ["coming out", "gender identity", "transition"],
            "exclude": []
        },
        "Substance Abuse": {
            "primary": ["substance", "addiction", "alcohol", "drug", "addict"],
            "secondary": ["recovery", "sobriety", "relapse", "withdrawal"],
            "exclude": []
        },
        "Eating Disorders": {
            "primary": ["eating disorder", "anorexia", "bulimia", "binge"],
            "secondary": ["body image", "weight obsession", "purge"],
            "exclude": []
        },
        "Sleep": {
            "primary": ["sleep", "insomnia", "nightmare", "sleep disorder"],
            "secondary": ["fatigue", "tired", "rest", "circadian"],
            "exclude": ["sleep it off"]  # Not about sleep problems
        },
        "Workplace": {
            "primary": ["work", "job", "career", "boss", "colleague"],
            "secondary": ["office", "employment", "profession", "coworker"],
            "exclude": ["homework", "housework"]  # Not workplace
        },
        "Behavioral Change": {
            "primary": ["behavior change", "habit", "pattern", "routine"],
            "secondary": ["motivation", "goal", "self improvement"],
            "exclude": []
        },
    }
    
    return label_patterns

def improved_topic_mapping(topic_str, label_patterns):
    """More sophisticated topic to label mapping"""
    
    if not isinstance(topic_str, str):
        return ["Other"]
    
    topic_lower = topic_str.lower()
    matched_labels = set()
    scores = {}  # Track confidence scores
    
    # Check each label pattern
    for label, patterns in label_patterns.items():
        score = 0
        
        # Check exclusions first
        excluded = False
        for exclude_term in patterns.get("exclude", []):
            if exclude_term in topic_lower:
                excluded = True
                break
        
        if excluded:
            continue
        
        # Primary keywords (higher weight)
        for keyword in patterns["primary"]:
            if keyword in topic_lower:
                score += 2
        
        # Secondary keywords (lower weight)
        for keyword in patterns.get("secondary", []):
            if keyword in topic_lower:
                score += 1
        
        if score > 0:
            scores[label] = score
    
    # Select labels based on scores
    if scores:
        # Get labels with score >= 2, or top label if all scores are 1
        max_score = max(scores.values())
        if max_score >= 2:
            matched_labels = {label for label, score in scores.items() if score >= 2}
        else:
            # Take all labels with the max score
            matched_labels = {label for label, score in scores.items() if score == max_score}
    
    # Handle compound topics
    if "depression" in topic_lower and "anxiety" in topic_lower:
        matched_labels.update(["Depression", "Anxiety"])
    
    if "relationship" in topic_lower and "family" in topic_lower:
        matched_labels.add("Family Conflict")
        matched_labels.discard("Relationships")  # Remove general relationships
    
    # Fallback
    if not matched_labels:
        # Try to find any relevant mental health keyword
        mental_health_keywords = [
            "mental", "emotional", "psychological", "therapy", 
            "counseling", "psychiatric", "wellness"
        ]
        
        if any(keyword in topic_lower for keyword in mental_health_keywords):
            matched_labels.add("Counseling")
        else:
            matched_labels.add("Other")
    
    return list(matched_labels)

def validate_labels(df, label_patterns):
    """Validate and clean the labeled data"""
    
    print("Validating labels...")
    
    # Apply improved mapping
    df["improved_labels"] = df["topics"].apply(
        lambda x: improved_topic_mapping(x, label_patterns)
    )
    
    # Compare with original labels
    changes = 0
    for idx, row in df.iterrows():
        original = set(eval(row["mapped_labels"]) if pd.notna(row["mapped_labels"]) else [])
        improved = set(row["improved_labels"])
        
        if original != improved:
            changes += 1
            if changes <= 5:  # Show first 5 changes
                print(f"\nExample change:")
                print(f"  Topic: {row['topics']}")
                print(f"  Original: {original}")
                print(f"  Improved: {improved}")
    
    print(f"\nTotal changes: {changes} out of {len(df)} ({changes/len(df)*100:.1f}%)")
    
    # Analyze label distribution
    all_labels = [label for labels in df["improved_labels"] for label in labels]
    label_counts = Counter(all_labels)
    
    print("\nImproved label distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    # Check for problematic patterns
    print("\nQuality checks:")
    
    # Check single-label dominance
    single_label_count = sum(1 for labels in df["improved_labels"] if len(labels) == 1)
    print(f"  Single-label samples: {single_label_count} ({single_label_count/len(df)*100:.1f}%)")
    
    # Check "Other" usage
    other_count = sum(1 for labels in df["improved_labels"] if "Other" in labels)
    print(f"  'Other' label usage: {other_count} ({other_count/len(df)*100:.1f}%)")
    
    # Check label co-occurrence
    cooccurrence = {}
    for labels in df["improved_labels"]:
        for i, l1 in enumerate(labels):
            for l2 in labels[i+1:]:
                pair = tuple(sorted([l1, l2]))
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
    
    print("\nTop label co-occurrences:")
    for pair, count in sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pair[0]} + {pair[1]}: {count}")
    
    return df

# Additional utility functions

def merge_rare_labels(df, min_samples=30):
    """Merge rare labels into related categories or 'Other'"""
    
    # Count label frequencies
    label_counts = Counter()
    for labels in df["improved_labels"]:
        for label in labels:
            label_counts[label] += 1
    
    # Define merge rules for rare labels
    merge_rules = {
        "Military": "Trauma",  # Military issues often involve trauma
        "Spirituality": "Counseling",  # General counseling
        "Professional Ethics": "Workplace",  # Work-related
        "Diagnosis": "Counseling",  # General mental health
    }
    
    # Apply merging
    def merge_labels(label_list):
        merged = set()
        for label in label_list:
            if label_counts[label] < min_samples and label in merge_rules:
                merged.add(merge_rules[label])
            elif label_counts[label] < min_samples:
                merged.add("Other")
            else:
                merged.add(label)
        return list(merged)
    
    df["merged_labels"] = df["improved_labels"].apply(merge_labels)
    
    print(f"\nLabel merging results (min_samples={min_samples}):")
    for old_label, new_label in merge_rules.items():
        if label_counts[old_label] < min_samples:
            print(f"  {old_label} ({label_counts[old_label]}) → {new_label}")
    
    return df
def strip_relationship_prefix(topic_str):
    if pd.isna(topic_str):
        return topic_str
    parts = [t.strip() for t in topic_str.split(",") if t.strip().lower() != "relationships"]
    return ", ".join(parts) if parts else ""

# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("counselchat_labels.csv")
    
    df["topics"] = df["topics"].apply(strip_relationship_prefix)
    
    # Create improved label patterns
    label_patterns = create_improved_label_mapping()
    
    # Apply validation and improvements
    df = validate_labels(df, label_patterns)
    
    # Optional: merge rare labels
    df = merge_rare_labels(df, min_samples=30)
    
    # Save improved labels
    df["mapped_labels"] = df["merged_labels"]  # or "improved_labels" if not merging
    df.to_csv("counselchat_labels_improved.csv", index=False)
    
    print("\nImproved labels saved to counselchat_labels_improved.csv")