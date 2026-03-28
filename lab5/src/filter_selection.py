import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

def apply_filter_pipeline(X_train, X_test, y_train):
    t0 = time.time()

    # Step 1: Variance filter
    sel = VarianceThreshold(threshold=0.01)
    sel.fit(X_train)
    cols = X_train.columns[sel.get_support()]
    X_train, X_test = X_train[cols], X_test[cols]
    print(f"Variance filter:    → {X_train.shape[1]} features")

    # Step 2: Correlation filter
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.99)]
    X_train, X_test = X_train.drop(columns=to_drop), X_test.drop(columns=to_drop)
    print(f"Correlation filter: → {X_train.shape[1]} features")

    # Step 3: Mutual information filter
    mi = mutual_info_regression(X_train, y_train, random_state=42)
    top_cols = pd.Series(mi, index=X_train.columns).nlargest(50).index
    X_train, X_test = X_train[top_cols], X_test[top_cols]
    print(f"MI filter:          → {X_train.shape[1]} features")

    print(f"Filter done in {time.time()-t0:.1f}s")
    return X_train, X_test

if __name__ == "__main__":
    feat_train = pd.read_parquet("../outputs/features_train.parquet")
    feat_test  = pd.read_parquet("../outputs/features_test.parquet")
    y_train    = pd.read_csv("../outputs/labels_train.csv", index_col=0).squeeze()

    X_train_f, X_test_f = apply_filter_pipeline(feat_train, feat_test, y_train)
    X_train_f.to_parquet("../outputs/filtered_train.parquet")
    X_test_f.to_parquet("../outputs/filtered_test.parquet")
    print("Saved.")