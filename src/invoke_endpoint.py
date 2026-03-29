import os
import json
import requests
import numpy as np
import pandas as pd
import urllib3
from sklearn.metrics import accuracy_score, f1_score

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ENDPOINT_URL = "https://amazon-review-endpoint.qatarcentral.inference.ml.azure.com/score"
API_KEY      = os.environ.get("AZURE_ML_API_KEY", "")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

NON_FEATURE_COLS = {
    "asin","reviewerID","overall","overall_x","overall_y","label",
    "review_year","review_month","review_date",
    "reviewText","summary","unixReviewTime","reviewTime"
}

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
    return np.hstack(feature_parts).astype(np.float32)

def main():
    print("Loading deploy dataset...")
    df = pd.read_parquet("data/deploy/azureml/5fb84b3b-6a46-4a1e-bf07-831e79ce9d5e/out/data.parquet")

    overall_cols = [c for c in df.columns if "overall" in c.lower()]
    if "overall" not in df.columns and overall_cols:
        df = df.rename(columns={overall_cols[0]: "overall"})
    df["label"] = (df["overall"] >= 4).astype(int)
    y_true = df["label"].values

    print("Building features...")
    X = build_features(df)
    # pad to match training feature count if needed
    expected_features = 891
    if X.shape[1] < expected_features:
        pad = np.zeros((X.shape[0], expected_features - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])

    print(f"Sending {len(X)} samples to endpoint in batches...")
    all_preds = []
    batch_size = 100

    session = requests.Session()
    session.verify = False

    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        payload = {"data": batch.tolist()}
        response = session.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload), timeout=60)
        if response.status_code != 200:
            print(f"Error batch {i}: {response.status_code} {response.text}")
            continue
        result = response.json()
        if "predictions" in result:
            all_preds.extend(result["predictions"])
        elif "result" in result:
            all_preds.extend(result["result"])
        else:
            print(f"Unexpected response format: {result}")
            break

    all_preds = np.array(all_preds)
    acc = accuracy_score(y_true[:len(all_preds)], all_preds)
    f1  = f1_score(y_true[:len(all_preds)], all_preds, zero_division=0)
    print(f"\n=== Deployment Results ===")
    print(f"Samples: {len(all_preds)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1: {f1:.4f}")

if __name__ == "__main__":
    main()