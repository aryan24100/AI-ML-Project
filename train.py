import json
import os
import pickle
import random
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nlp_utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load dataset
with open(INTENTS_PATH, encoding="utf-8") as file:
    data = json.load(file)


def augment_pattern(pattern):
    cleaned = preprocess_text(pattern)
    variants = {cleaned}

    prefixes = [
        "",
        "please ",
        "can you tell me ",
        "i want to know ",
        "give me details about ",
    ]
    suffixes = ["", " please", " in detail"]

    for prefix in prefixes:
        for suffix in suffixes:
            candidate = preprocess_text(f"{prefix}{cleaned}{suffix}")
            if candidate:
                variants.add(candidate)

    filler_words = {
        "what",
        "is",
        "the",
        "about",
        "please",
        "can",
        "you",
        "tell",
        "me",
        "for",
        "at",
    }
    compact_tokens = [token for token in cleaned.split() if token not in filler_words]
    if len(compact_tokens) >= 2:
        variants.add(" ".join(compact_tokens))

    if len(cleaned) > 14:
        variants.add(cleaned.replace(" please", ""))
        variants.add(cleaned.replace(" details", " info"))

    return list(variants)


texts = []
labels = []

# Prepare data
for intent in data['intents']:
    tag_text = preprocess_text(intent['tag'].replace("_", " "))
    if tag_text:
        texts.append(tag_text)
        labels.append(intent['tag'])
        texts.append(preprocess_text(f"{tag_text} details"))
        labels.append(intent['tag'])

    for pattern in intent['patterns']:
        for variant in augment_pattern(pattern):
            texts.append(variant)
            labels.append(intent['tag'])

# Shuffle once for stable but mixed training order
combined = list(zip(texts, labels))
random.Random(42).shuffle(combined)
texts, labels = zip(*combined)

# Vectorization (word + character n-grams)
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    strip_accents="unicode",
)
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True,
)

word_X = word_vectorizer.fit_transform(texts)
char_X = char_vectorizer.fit_transform(texts)
X = hstack([word_X, char_X]).tocsr()

# Model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
model.fit(X, labels)

# Save model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)
with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump({"word": word_vectorizer, "char": char_vectorizer}, vectorizer_file)

print(f"Model trained successfully on {len(texts)} samples!")