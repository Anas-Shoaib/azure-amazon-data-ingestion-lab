import argparse
import os
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

def main():
    args = parse_args()
    df = load_folder(args.data)
    print(f"Loaded {len(df)} rows")

    df["review_length_words"] = df["reviewText"].apply(lambda x: len(str(x).split()))
    df["review_length_chars"] = df["reviewText"].apply(lambda x: len(str(x)))

    out_df = df[["asin", "reviewerID", "review_length_words", "review_length_chars"]]

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Length features done.")

if __name__ == "__main__":
    main()
