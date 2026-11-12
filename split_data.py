
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project root
BASE_DIR = Path(__file__).resolve().parent

# Input and output paths
INPUT_PATH = BASE_DIR / "data" / "cleaned_news.csv"
OUTPUT_DIR = BASE_DIR / "data" / "splits"

# Check input file
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {INPUT_PATH}\n"
        "Run prepare_data.py first."
    )

# Load cleaned data
df = pd.read_csv(INPUT_PATH)

# First split: 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Second split: 10% validation, 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save datasets
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Display results
print("TRAIN SIZE:", len(train_df))
print("VALIDATION SIZE:", len(val_df))
print("TEST SIZE:", len(test_df))

print("\nLABEL DISTRIBUTION:")

for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(f"\n{name}:")
    print(data["label"].value_counts(normalize=True))

# Verify no exact content overlap
train_content = set(train_df["content"])
val_content = set(val_df["content"])
test_content = set(test_df["content"])

print("\nCONTENT OVERLAP:")
print("Train-Val:", len(train_content & val_content))
print("Train-Test:", len(train_content & test_content))
print("Val-Test:", len(val_content & test_content))
