import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

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

    sia = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if not isinstance(text, str) or len(text) == 0:
            return {"pos": 0.0, "neg": 0.0, "neu": 0.0, "compound": 0.0}
        return sia.polarity_scores(text)

    print("Computing sentiment scores...")
    scores = df["reviewText"].apply(get_sentiment)

    df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

    output_df = df[["asin", "reviewerID", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]]

    os.makedirs(args.out, exist_ok=True)
    output_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Sentiment features done.")
    print(output_df.head())

if __name__ == "__main__":
    main()
