import argparse
import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

    sia = SentimentIntensityAnalyzer()

    def get_scores(text):
        if not isinstance(text, str) or len(text) == 0:
            return {"pos": 0.0, "neg": 0.0, "neu": 0.0, "compound": 0.0}
        return sia.polarity_scores(text)

    print("Computing sentiment scores...")
    scores = df["reviewText"].apply(get_scores)

    df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

    out_df = df[[
        "asin", "reviewerID",
        "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"
    ]]

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Sentiment features done.")

if __name__ == "__main__":
    main()
