import argparse
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

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

def load_parquet_folder(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    args = parse_args()

    train_df = load_parquet_folder(args.train)
    val_df = load_parquet_folder(args.val)
    test_df = load_parquet_folder(args.test)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Fit ONLY on training data
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    train_text = train_df["reviewText"].fillna("").tolist()
    val_text = val_df["reviewText"].fillna("").tolist()
    test_text = test_df["reviewText"].fillna("").tolist()

    print("Fitting TF-IDF on training data...")
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)

    feature_names = [f"tfidf_{i}" for i in range(train_tfidf.shape[1])]

    def sparse_to_df(matrix, keys_df, feature_names):
        dense = pd.DataFrame(matrix.toarray(), columns=feature_names)
        keys = keys_df[["asin", "reviewerID"]].reset_index(drop=True)
        return pd.concat([keys, dense], axis=1)

    train_out_df = sparse_to_df(train_tfidf, train_df, feature_names)
    val_out_df = sparse_to_df(val_tfidf, val_df, feature_names)
    test_out_df = sparse_to_df(test_tfidf, test_df, feature_names)

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)

    train_out_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_out_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_out_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)

    print("TF-IDF features done.")

if __name__ == "__main__":
    main()
