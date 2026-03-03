import argparse
import os
import pandas as pd

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

def main():
    args = parse_args()

    files = list_parquet_files(args.data)
    if not files:
        raise ValueError(f"No parquet files found under: {args.data}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Feature Store requires a timestamp column
    df["event_timestamp"] = pd.Timestamp.utcnow()

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Wrote timestamped parquet:", os.path.join(args.out, "data.parquet"))
    print("Rows:", len(df), "Cols:", df.shape[1])

if __name__ == "__main__":
    main()
