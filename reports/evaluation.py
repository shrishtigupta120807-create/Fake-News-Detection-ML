
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# 1. PATHS
# =====================================

os.makedirs("reports/figures", exist_ok=True)

# =====================================
# 2. LOAD TEST DATA AND MODEL
# =====================================

test = pd.read_csv(
    "data/splits/test.csv",
    usecols=["content", "label"]
)

X_test = test["content"]
y_test = test["label"]

model = joblib.load("model/fake_news_model_clean.pkl")

# =====================================
# 3. PREDICTIONS
# =====================================

predictions = model.predict(X_test)

# =====================================
# 4. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Clean Model")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png", dpi=300)
plt.close()

# =====================================
# 5. CLASSIFICATION REPORT
# =====================================

report = classification_report(
    y_test,
    predictions,
    target_names=["Real", "Fake"],
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "reports/classification_report.csv"
)

# =====================================
# 6. MODEL METRICS
# =====================================

original = {
    "Accuracy": 94.93,
    "Fake Precision": 94.67,
    "Fake Recall": 94.00,
    "Fake F1": 94.33
}

masked = {
    "Accuracy": 94.60,
    "Fake Precision": 94.53,
    "Fake Recall": 93.36,
    "Fake F1": 93.94
}

comparison = pd.DataFrame({
    "Original Model": original,
    "Publisher-Masked Model": masked
})

comparison.to_csv("reports/model_comparison.csv")

# =====================================
# 7. COMPARISON CHART
# =====================================

comparison.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(90, 100),
    rot=0
)

plt.ylabel("Score (%)")
plt.title("Original vs Publisher-Masked Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/figures/model_comparison.png", dpi=300)
plt.close()

print("Evaluation completed successfully!")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(pd.DataFrame(report).transpose())
print("\nSaved all evaluation files inside reports/")
