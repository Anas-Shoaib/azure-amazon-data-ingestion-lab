import argparse
import os
import re
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Replace numbers with token
    text = re.sub(r'\d+', ' NUM ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Trim extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    args = parse_args()

    files = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(".parquet")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows")

    # Normalize reviewText
    df["reviewText"] = df["reviewText"].apply(normalize_text)

    # Filter out empty or very short reviews
    df = df[df["reviewText"].str.len() >= 10]
    print(f"After filtering: {len(df)} rows")

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Normalization complete.")

if __name__ == "__main__":
    main()
