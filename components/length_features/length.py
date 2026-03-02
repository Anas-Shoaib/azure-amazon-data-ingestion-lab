
import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    files = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(".parquet")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows")
    df["review_length_words"] = df["reviewText"].apply(lambda x: len(str(x).split()))
    df["review_length_chars"] = df["reviewText"].apply(lambda x: len(str(x)))
    output_df = df[["asin", "reviewerID", "review_length_words", "review_length_chars"]]
    os.makedirs(args.out, exist_ok=True)
    output_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Length features done.")

if __name__ == "__main__":
    main()

