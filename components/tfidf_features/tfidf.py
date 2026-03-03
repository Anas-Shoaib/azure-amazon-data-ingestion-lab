import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--test", type=str, required=True)
    p.add_argument("--train_out", type=str, required=True)
    p.add_argument("--val_out", type=str, required=True)
    p.add_argument("--test_out", type=str, required=True)
    p.add_argument("--max_features", type=int, default=500)
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
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[["asin", "reviewerID", "reviewText"]]

def save_out(df, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    df.to_parquet(os.path.join(out_folder, "data.parquet"), index=False)

def main():
    args = parse_args()

    train_df = load_folder(args.train)
    val_df = load_folder(args.val)
    test_df = load_folder(args.test)

    print("Train rows:", len(train_df), "Val rows:", len(val_df), "Test rows:", len(test_df))
    print("max_features =", args.max_features)

    vec = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train = vec.fit_transform(train_df["reviewText"].fillna("").tolist()).astype(np.float32)
    X_val = vec.transform(val_df["reviewText"].fillna("").tolist()).astype(np.float32)
    X_test = vec.transform(test_df["reviewText"].fillna("").tolist()).astype(np.float32)

    print("TFIDF shapes:", X_train.shape, X_val.shape, X_test.shape)

    cols = [f"tfidf_{i}" for i in range(X_train.shape[1])]

    def build(keys_df, X):
        dense = pd.DataFrame(X.toarray(), columns=cols, dtype="float32")
        keys = keys_df[["asin", "reviewerID"]].reset_index(drop=True)
        return pd.concat([keys, dense], axis=1)

    save_out(build(train_df, X_train), args.train_out)
    save_out(build(val_df, X_val), args.val_out)
    save_out(build(test_df, X_test), args.test_out)

    print("TF-IDF done.")

if __name__ == "__main__":
    main()
