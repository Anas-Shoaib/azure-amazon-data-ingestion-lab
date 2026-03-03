import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--train_out", type=str, required=True)
    p.add_argument("--val_out", type=str, required=True)
    p.add_argument("--test_out", type=str, required=True)
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

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - args.train_ratio),
        random_state=args.seed,
        shuffle=True
    )

    val_size = args.val_ratio / (1 - args.train_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=args.seed,
        shuffle=True
    )

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))

if __name__ == "__main__":
    main()
