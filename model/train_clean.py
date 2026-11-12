
import os
import joblib
import pandas as pd

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

# 1. LOAD EXISTING SPLITS

print("Loading datasets...")

train_df = pd.read_csv(
    "data/splits/train.csv",
    usecols=["content", "label"]
)

val_df = pd.read_csv(
    "data/splits/validation.csv",
    usecols=["content", "label"]
)

test_df = pd.read_csv(
    "data/splits/test.csv",
    usecols=["content", "label"]
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# 2. VERIFY LABELS

for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(f"\n{name} label distribution:")
    print(data["label"].value_counts())

# 3. BUILD PIPELINE

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

# 4. TRAIN

print("\nTraining model...")

model.fit(
    train_df["content"],
    train_df["label"]
)

print("Training completed!")

# 5. EVALUATION FUNCTION

def evaluate(data, split_name):

    y_true = data["label"]
    y_pred = model.predict(data["content"])

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    print(f"\n========== {split_name.upper()} RESULTS ==========")

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    return {
        "Split": split_name,
        "Accuracy": accuracy,
        "Precision_Fake": precision,
        "Recall_Fake": recall,
        "F1_Fake": f1
    }

# 6. VALIDATION AND FINAL TEST

val_metrics = evaluate(val_df, "Validation")

test_metrics = evaluate(test_df, "Test")

# 7. SAVE MODEL AND METRICS

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/fake_news_model_clean.pkl"
)

pd.DataFrame(
    [val_metrics, test_metrics]
).to_csv(
    "model/clean_metrics.csv",
    index=False
)

print("\nClean model and metrics saved successfully!")
