import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
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

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["reviewText"].fillna("").tolist()
    print(f"Encoding {len(texts)} reviews...")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embed_cols = [f"bert_embedding_{i}" for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)

    keys = df[["asin", "reviewerID"]].reset_index(drop=True)
    out_df = pd.concat([keys, embed_df], axis=1)

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("SBERT embeddings done.")

if __name__ == "__main__":
    main()
