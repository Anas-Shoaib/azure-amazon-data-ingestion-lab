import time
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from load_data import load_cmapss, preprocess

def prepare_for_tsfresh(df):
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    op_cols     = [c for c in df.columns if c.startswith("op_setting_")]
    keep_cols   = ["engine_id", "cycle", "RUL"] + sensor_cols + op_cols

    df = df[keep_cols].sort_values(["engine_id", "cycle"]).copy()

    sliced = []
    for eid, group in df.groupby("engine_id"):
        sliced.append(group.iloc[:max(1, int(len(group)*0.7))])
    df = pd.concat(sliced).reset_index(drop=True)

    labels = df.groupby("engine_id")["RUL"].last()
    labels.index.name = "id"

    ts_df = df[["engine_id", "cycle"] + sensor_cols + op_cols].copy()
    ts_df = ts_df.rename(columns={"engine_id": "id", "cycle": "time"})

    return ts_df, labels

def extract(df, fast=True):
    ts_df, labels = prepare_for_tsfresh(df)
    t0 = time.time()
    print("Extracting features...")
    features = extract_features(
        ts_df, column_id="id", column_sort="time",
        default_fc_parameters={
            "mean": None,
            "standard_deviation": None,
            "maximum": None,
            "minimum": None,
            "median": None,
        },
        n_jobs=1, show_warnings=False, disable_progressbar=False
    )
    impute(features)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — shape: {features.shape}")
    return features, labels, elapsed

if __name__ == "__main__":
    train_df, test_df = load_cmapss('../data/train_FD001.txt','../data/test_FD001.txt','../data/RUL_FD001.txt')
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)

    feat_train, y_train, t1 = extract(train_df)
    feat_test,  y_test,  t2 = extract(test_df)

    feat_train.to_parquet("../outputs/features_train.parquet")
    feat_test.to_parquet("../outputs/features_test.parquet")
    y_train.to_csv("../outputs/labels_train.csv")
    y_test.to_csv("../outputs/labels_test.csv")
    print("Saved to outputs/")