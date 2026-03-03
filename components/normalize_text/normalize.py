import argparse
import os
import re
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()

def list_parquet_files(folder):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.endswith(".parquet"):
                files.append(os.path.join(root, n))
    return files

def load_folder(folder):
    files = list_parquet_files(folder)
    if len(files) == 0:
        raise ValueError(f"No parquet files found under: {folder}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"\d+", " NUM ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    args = parse_args()
    df = load_folder(args.data)
    print(f"Loaded {len(df)} rows")

    df["reviewText"] = df["reviewText"].apply(normalize_text)
    df = df[df["reviewText"].str.len() >= 10]
    print(f"After filtering: {len(df)} rows")

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Normalization complete.")

if __name__ == "__main__":
    main()
