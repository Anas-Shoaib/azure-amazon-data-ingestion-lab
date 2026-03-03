import argparse
import os
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=str, required=True)
    p.add_argument("--sentiment", type=str, required=True)
    p.add_argument("--tfidf", type=str, required=True)
    p.add_argument("--sbert", type=str, required=True)
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

def main():
    args = parse_args()
    print("Loading all feature datasets...")

    length_df = load_folder(args.length)
    sentiment_df = load_folder(args.sentiment)
    tfidf_df = load_folder(args.tfidf)
    sbert_df = load_folder(args.sbert)

    keys = ["asin", "reviewerID"]

    print("Merging...")
    merged = length_df.merge(sentiment_df, on=keys, how="inner")
    merged = merged.merge(tfidf_df, on=keys, how="inner")
    merged = merged.merge(sbert_df, on=keys, how="inner")

    print(f"Merged shape: {merged.shape}")

    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Merge complete.")

if __name__ == "__main__":
    main()
