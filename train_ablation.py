
import re
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project paths
ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"

TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

# Check input files
for path in [TRAIN_PATH, TEST_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required split not found: {path}\n"
            "Run prepare_data.py and split_data.py first."
        )

# Load existing splits
train = pd.read_csv(TRAIN_PATH, usecols=["content", "label"])
test = pd.read_csv(TEST_PATH, usecols=["content", "label"])

# Publishers and source names to mask
publishers = [
    "Reuters",
    "Breitbart",
    "New York Times",
    "The New York Times",
    "Washington Post",
    "CNN",
    "Fox News",
    "BBC",
    "The Guardian",
    "Associated Press",
    "AP News",
    "USA Today",
    "Wall Street Journal",
    "Huffington Post",
    "Daily Mail",
    "The Independent"
]

# Build regex pattern
pattern = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(p)
        for p in sorted(publishers, key=len, reverse=True)
    )
    + r")\b",
    flags=re.IGNORECASE
)


def mask_publishers(text):
    return pattern.sub(" PUBLISHER ", str(text))


# Create masked content
print("Masking publisher names...")

X_train = train["content"].apply(mask_publishers)
X_test = test["content"].apply(mask_publishers)

y_train = train["label"]
y_test = test["label"]

# Build model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train
print("Training publisher-masked model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nPUBLISHER-ABLATED TEST RESULTS")
print("--------------------------------")

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2))
print("Fake Precision:", round(precision_score(y_test, predictions, pos_label=1) * 100, 2))
print("Fake Recall:", round(recall_score(y_test, predictions, pos_label=1) * 100, 2))
print("Fake F1:", round(f1_score(y_test, predictions, pos_label=1) * 100, 2))
