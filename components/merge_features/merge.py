import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def load_parquet_folder(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    args = parse_args()

    print("Loading feature datasets...")
    length_df = load_parquet_folder(args.length)
    sentiment_df = load_parquet_folder(args.sentiment)
    tfidf_df = load_parquet_folder(args.tfidf)
    sbert_df = load_parquet_folder(args.sbert)

    keys = ["asin", "reviewerID"]

    print("Merging all features on asin + reviewerID...")
    merged = length_df.merge(sentiment_df, on=keys, how="inner")
    merged = merged.merge(tfidf_df, on=keys, how="inner")
    merged = merged.merge(sbert_df, on=keys, how="inner")

    print(f"Merged shape: {merged.shape}")

    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Merge complete.")

if __name__ == "__main__":
    main()
