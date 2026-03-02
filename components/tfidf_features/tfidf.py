
import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=5000)
    return parser.parse_args()

def load_folder(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    args = parse_args()
    train_df = load_folder(args.train)
    val_df = load_folder(args.val)
    test_df = load_folder(args.test)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    vectorizer = TfidfVectorizer(max_features=args.max_features, stop_words="english", ngram_range=(1, 2))
    print("Fitting TF-IDF on training data only...")
    train_tfidf = vectorizer.fit_transform(train_df["reviewText"].fillna("").tolist())
    val_tfidf = vectorizer.transform(val_df["reviewText"].fillna("").tolist())
    test_tfidf = vectorizer.transform(test_df["reviewText"].fillna("").tolist())
    feature_names = [f"tfidf_{i}" for i in range(train_tfidf.shape[1])]
    def to_df(matrix, keys_df):
        dense = pd.DataFrame(matrix.toarray(), columns=feature_names)
        keys = keys_df[["asin", "reviewerID"]].reset_index(drop=True)
        return pd.concat([keys, dense], axis=1)
    for out_path, matrix, source_df in [
        (args.train_out, train_tfidf, train_df),
        (args.val_out, val_tfidf, val_df),
        (args.test_out, test_tfidf, test_df)
    ]:
        os.makedirs(out_path, exist_ok=True)
        to_df(matrix, source_df).to_parquet(os.path.join(out_path, "data.parquet"), index=False)
    print("TF-IDF features done.")

if __name__ == "__main__":
    main()

