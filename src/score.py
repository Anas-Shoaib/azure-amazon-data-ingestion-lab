import json
import os
import joblib
import numpy as np
import pandas as pd

model = None

NON_FEATURE_COLS = {
    "asin", "reviewerID", "overall", "overall_x", "overall_y", "label",
    "review_year", "review_month", "review_date",
    "reviewText", "summary", "unixReviewTime", "reviewTime"
}

def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    # model is stored in model_output subfolder
    model_path = os.path.join(model_dir, "model_output", "model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

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

def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = np.array(data["data"], dtype=np.float32)
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1].tolist()
        return {"predictions": preds.tolist(), "probabilities": proba}
    except Exception as e:
        return {"error": str(e)}