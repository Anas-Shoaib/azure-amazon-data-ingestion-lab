import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_out",  type=str, required=True)
    p.add_argument("--val_out",    type=str, required=True)
    p.add_argument("--test_out",   type=str, required=True)
    p.add_argument("--deploy_out", type=str, required=True)
    return p.parse_args()

def list_parquet_files(folder):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.endswith(".parquet"):
                files.append(os.path.join(root, n))
    return files

def main():
    args = parse_args()
    files = list_parquet_files(args.data)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows")

    # Deploy split = most recent 10% by review_year
    if "review_year" in df.columns:
        df = df.sort_values("review_year").reset_index(drop=True)
    n = len(df)
    deploy_df = df.iloc[int(n * 0.90):]
    rest_df   = df.iloc[:int(n * 0.90)]

    # Split rest into train 60%, val 15%, test 15% (of total = 67/17/17 of rest)
    train_df, temp_df = train_test_split(rest_df, test_size=0.333, random_state=args.seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5,   random_state=args.seed)

    for out, split_df, name in [
        (args.train_out,  train_df,  "train"),
        (args.val_out,    val_df,    "val"),
        (args.test_out,   test_df,   "test"),
        (args.deploy_out, deploy_df, "deploy"),
    ]:
        os.makedirs(out, exist_ok=True)
        split_df.to_parquet(os.path.join(out, "data.parquet"), index=False)
        print(f"{name} rows: {len(split_df)}")

if __name__ == "__main__":
    main()
