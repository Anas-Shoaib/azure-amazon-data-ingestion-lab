import argparse
import os
import time
import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--test_data",  type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--C",        type=float, default=1.0)
    parser.add_argument("--max_iter", type=int,   default=1000)
    return parser.parse_args()

NON_FEATURE_COLS = {
    "asin","reviewerID","overall","label",
    "review_year","review_month","review_date",
    "reviewText","summary","unixReviewTime","reviewTime"
}

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return pd.read_parquet(path)

def create_labels(df):
    # Handle column name variations from merge
    if "overall" not in df.columns:
        overall_cols = [c for c in df.columns if "overall" in c.lower()]
        if overall_cols:
            df = df.rename(columns={overall_cols[0]: "overall"})
        else:
            raise RuntimeError(f"No overall column found. Columns: {list(df.columns)}")
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df):
    feature_parts = []

    if "sbert_vector" in df.columns:
        feature_parts.append(np.vstack(df["sbert_vector"].values))
    else:
        sbert_cols = sorted([c for c in df.columns if c.startswith("sbert_")],
                            key=lambda x: int(x.split("_")[1]))
        if sbert_cols:
            feature_parts.append(df[sbert_cols].values)

    tfidf_cols = sorted([c for c in df.columns if c.startswith("tfidf_")])
    if tfidf_cols:
        feature_parts.append(df[tfidf_cols].values)

    sentiment_cols = [c for c in df.columns if "sentiment" in c.lower()]
    if sentiment_cols:
        feature_parts.append(df[sentiment_cols].values)

    length_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith("sbert_")
        and not c.startswith("tfidf_")
        and "sentiment" not in c.lower()
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    if length_cols:
        feature_parts.append(df[length_cols].values)

    if not feature_parts:
        raise RuntimeError("No feature columns found.")

    return np.hstack(feature_parts).astype(np.float32)

def evaluate(model, X, y, split):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc  = accuracy_score(y, preds)
    auc  = roc_auc_score(y, proba)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    mlflow.log_metric(f"{split}_accuracy",  acc)
    mlflow.log_metric(f"{split}_auc",       auc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall",    rec)
    mlflow.log_metric(f"{split}_f1",        f1)
    print(f"[{split}] acc={acc:.4f} auc={auc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

def main():
    args = parse_args()
    start_time = time.time()
    mlflow.start_run()
    mlflow.log_param("C",        args.C)
    mlflow.log_param("max_iter", args.max_iter)

    print("Loading data...")
    train_df = create_labels(load_data(args.train_data))
    val_df   = create_labels(load_data(args.val_data))
    test_df  = create_labels(load_data(args.test_data))

    print("Building features...")
    X_train = build_features(train_df)
    X_val   = build_features(val_df)
    X_test  = build_features(test_df)
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    print(f"Train shape: {X_train.shape}")
    print("Training model...")
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="saga", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val,   y_val,   "val")
    evaluate(model, X_test,  y_test,  "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Done in {runtime:.1f}s")
    mlflow.end_run()

if __name__ == "__main__":
    main()