
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(r"C:\Users\rahul\Downloads\WELFake_Dataset.csv")

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]

print("Original dataset shape:", df.shape)


# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

print("\nPreprocessing data...")

# Remove rows without labels
df = df.dropna(subset=["label"]).copy()

# Handle missing title and text
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine title and article
df["content"] = df["title"] + " " + df["text"]

# Remove empty articles
df = df[df["content"].str.strip() != ""]

# Convert labels to integers
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

# Verify label values
if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels! Expected 0 and 1.")

# Remove duplicate content
df = df.drop_duplicates(subset=["content"])

print("Cleaned dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ==========================================
# 3. DATA SAMPLING
# ==========================================

# Use a balanced sample for faster training.
# Set USE_SAMPLE = False to train on all available data.

USE_SAMPLE = True

if USE_SAMPLE:

    samples_per_class = min(
        15000,
        df["label"].value_counts().min()
    )

    df = (
        df.groupby("label", group_keys=False)
        .sample(n=samples_per_class, random_state=42)
    )

print("\nDataset used for training:", df.shape)


# ==========================================
# 4. SEPARATE FEATURES AND LABELS
# ==========================================

X = df["content"]
y = df["label"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. BUILD MACHINE LEARNING PIPELINE
# ==========================================

print("\nBuilding ML model...")

model = Pipeline([

    # Convert text into numerical features
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

    # Classification algorithm
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

print("\nTesting model...")

y_pred = model.predict(X_test)


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX:")
print(cm)


# ==========================================
# 10. SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/fake_news_model.pkl"
)

print("\nModel saved successfully!")

# Save performance metrics for the project report
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

pd.DataFrame([metrics]).to_csv(
    "model/metrics.csv",
    index=False
)

print("Metrics saved successfully!")

print("\nALL TASKS COMPLETED!")
