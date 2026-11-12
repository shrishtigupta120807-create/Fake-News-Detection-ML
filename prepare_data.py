
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = ROOT / "data" / "raw" / "WELFake_Dataset.csv"
OUTPUT_PATH = ROOT / "data" / "cleaned_news.csv"

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_DATA_PATH}\n"
        "Place WELFake_Dataset.csv inside data/raw/"
    )

df = pd.read_csv(RAW_DATA_PATH)

print("Original shape:", df.shape)

df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
df.dropna(subset=["text"], inplace=True)

df["title"] = df["title"].fillna("")
df["title"] = df["title"].str.strip()
df["text"] = df["text"].str.strip()

df = df[df["text"] != ""].copy()

df["content"] = df["title"] + " " + df["text"]

df.drop_duplicates(subset=["content"], inplace=True)
df.reset_index(drop=True, inplace=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nCLEANING COMPLETED")
print("Final shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nOriginal rows:", 72134)
print("After removing missing text:", 72095)
print("After removing duplicates:", len(df))
