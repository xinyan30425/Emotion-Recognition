# 2. assign_labels.py
import pandas as pd
import re
from collections import defaultdict

# ---------- 1) Emotion schema (tighter "down" rules) ----------
EMOTION_KEYWORDS = {
    "Anger": [
        r"\bangry\b", r"\banger\b", r"\bfurious\b|\brage\b",
        r"\birritat(?:ed|ing|ion)\b", r"\bannoy(?:ed|ing|ance)\b",
        r"\bresent(?:ful|ment)\b", r"\bsnapp(?:y|ing)\b",
        r"\blost (?:my|your|his|her) temper\b",
    ],
    "Anxiety": [
        r"\banxiety\b|\banxious\b", r"\bpanic (?:attack|attacks)\b", r"\bpanic(?:king)?\b",
        r"\bworr(?:y|ied|ying)\b", r"\bnervous(?:|ness)\b", r"\bon edge\b",
        r"\brestless(?:|ness)\b", r"\boverwhelm(?:ed|ing)\b",
        r"\bheart (?:is )?racing\b", r"\bcatastroph(?:e|izing)\b", r"\bfear(?:ful|s)?\b",
    ],
    "Sadness/Depression": [
        r"\bdepress(?:ed|ion)\b",
        r"\b(feel(?:ing)?|felt|feels|am|is|are|was|were|seem(?:s|ed)?)\s+(?:so\s+|very\s+|really\s+)?down\b",
        r"\bdown about\b", r"\bdown lately\b", r"\bfeeling blue\b", r"\blow mood\b",
        r"\bhopeless(?:|ness)\b", r"\bworthless(?:|ness)\b", r"\bempt(?:y|iness)\b",
        r"\banhedonia\b|loss of interest|no pleasure",
        r"\btearful\b|\bcry(?:ing)?\b",
        r"\bno energy\b|\bexhaust(?:ed|ion)\b|\bfatigue\b",
        r"\bcan(?:no|')t get out of bed\b",
    ],
    "Joy/Positive": [
        r"\bhappy\b|\bhappiness\b|\bjoy(?:ful)?\b", r"\bexcited\b|\bthrilled\b",
        r"\bgrateful\b|\bthankful\b", r"\bcontent\b|\bsatisfied\b|\bat peace\b",
        r"\bproud of (?:myself|him|her|them)\b",
    ],
    "Fear": [
        r"\bscared\b|\bafraid\b|\bterrified\b|\bfrightened\b",
        r"\bfear of [^\s]+\b", r"\bphobia\b", r"\bdread\b", r"\bunsafe\b|\bin danger\b",
    ],
    "Disgust": [
        r"\bdisgust(?:ed|ing)\b", r"\brepuls(?:e|ed|ive)\b", r"\bgross(?:ed out)?\b",
    ],
    "Surprise": [
        r"\bsurpris(?:e|ed|ing)\b", r"\bshocked\b|\bastonish(?:ed|ing)\b", r"\bdidn'?t expect\b",
    ],
    "Guilt/Shame": [
        r"\bguilt(?:y|iness)\b", r"\bashamed\b|\bshame\b|\bembarrass(?:ed|ment)\b",
        r"\bregret(?:|ful)\b",
    ],
    "Grief": [
        r"\bgrief\b|\bbereave(?:d|ment)\b",
        r"\blost (?:my|our) (?:mom|dad|parent|child|friend|partner)\b",
        r"\bpassed away\b|\bfuneral\b",
    ],
    "Loneliness": [
        r"\blonely\b|\balone\b|\bisolated\b|\bisolation\b",
        r"\bno (?:friends|support|one to talk)\b",
    ],
    "Stress/Tension": [
        r"\bstress(?:|ed|ful|ors?)\b", r"\bburnout\b",
        r"\btoo much on (?:my|your|his|her) plate\b",
        r"\bcan(?:no|')t cope\b|\boverwhelmed\b", r"\btense\b|\btension\b",
    ],
}

# ---------- 2) Exclusions for ambiguous phrases (esp. 'down') ----------
EXCLUDE_PATTERNS = {
    "Sadness/Depression": re.compile(
        r"\b("
        r"lie down|lay down|laid down|sit down|calm down|slow down|"
        r"shut ?down|shutdown|break down|breakdown|"
        r"go down|went down|gone down|fall(?:ing)? down|"
        r"down the\b|down to\b|down on\b|down there\b|downstairs\b|"
        r"download|downhill|downsize|down payment"
        r")\b",
        flags=re.IGNORECASE,
    ),
    # You can add other label-specific excludes here if needed.
}

# ---------- 3) Negation pattern ----------
NEGATIONS = r"\b(no|not|never|without|none|hardly|barely|rarely|den(y|ies|ied)|don'?t|doesn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|can'?t|couldn'?t|shouldn'?t|won'?t)\b"

def compile_patterns(schema):
    compiled = []
    for label, patterns in schema.items():
        for p in patterns:
            # treat '-' and space the same (optional, kept from your version)
            p_norm = p.replace(r"-", r"[- ]")
            compiled.append((re.compile(p_norm, re.IGNORECASE), label))
    return compiled

PATTERNS = compile_patterns(EMOTION_KEYWORDS)

def is_negated(text, match_start, window_chars=30):
    start = max(0, match_start - window_chars)
    context = text[start:match_start]
    return re.search(NEGATIONS, context, flags=re.IGNORECASE) is not None

def violates_exclusion(label, text, window_start, window_end):
    """
    For a matched label span, check a local window for exclude phrases.
    """
    ex = EXCLUDE_PATTERNS.get(label)
    if not ex:
        return False
    window = text[max(0, window_start-20): min(len(text), window_end+20)]
    return ex.search(window) is not None

# ---------- 4) Labeling with context-aware checks ----------
def assign_emotion_labels(text):
    if not isinstance(text, str) or not text.strip():
        return [], {}
    hits = defaultdict(int)

    for pat, label in PATTERNS:
        for m in pat.finditer(text):
            # Negation guard
            if is_negated(text, m.start()):
                continue
            # Exclusion guard (esp. for 'down' phrases)
            if violates_exclusion(label, text, m.start(), m.end()):
                continue
            hits[label] += 1

    labels = sorted(hits.keys())
    return labels, dict(hits)

# ---------- 5) CLI ----------
if __name__ == "__main__":
    in_path = "merged_transcripts.csv"
    out_path = "labeled_transcripts_emotions.csv"

    df = pd.read_csv(in_path)
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in merged_transcripts.csv")

    results = df["text"].apply(assign_emotion_labels)
    df["emotion_labels"] = results.apply(lambda x: x[0])
    df["emotion_hits"]   = results.apply(lambda x: x[1])

    df.to_csv(out_path, index=False)
    total = len(df)
    labeled = int((df["emotion_labels"].map(len) > 0).sum())
    print(f"Saved to {out_path} | labeled {labeled}/{total} ({labeled/total:.1%})")
