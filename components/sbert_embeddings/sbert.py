cat > sbert.py << 'EOF'
import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    files = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(".parquet")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["reviewText"].fillna("").tolist()
    print(f"Encoding {len(texts)} reviews...")
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    embed_cols = [f"bert_embedding_{i}" for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)
    keys = df[["asin", "reviewerID"]].reset_index(drop=True)
    output_df = pd.concat([keys, embed_df], axis=1)
    os.makedirs(args.out, exist_ok=True)
    output_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("SBERT embeddings done.")

if __name__ == "__main__":
    main()
EOF
