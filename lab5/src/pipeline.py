import time
import pandas as pd
from load_data import load_cmapss, preprocess
from extract_features import extract
from filter_selection import apply_filter_pipeline
from ga_selection import run_ga
from train_model import evaluate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

if __name__ == '__main__':
    total_start = time.time()

    print("\n=== STEP 1: Load & Preprocess ===")
    train_df, test_df = load_cmapss('../data/train_FD001.txt','../data/test_FD001.txt','../data/RUL_FD001.txt')
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)

    print("\n=== STEP 2: Feature Extraction ===")
    feat_train, y_train, t1 = extract(train_df)
    feat_test,  y_test,  t2 = extract(test_df)

    print("\n=== STEP 3: Filter Selection ===")
    feat_train_f, feat_test_f = apply_filter_pipeline(feat_train, feat_test, y_train)

    print("\n=== STEP 4: GA Selection ===")
    selected, t_ga = run_ga(feat_train_f, y_train)
    X_train = feat_train_f[selected]
    X_test  = feat_test_f[selected]

    print("\n=== STEP 5: Model Evaluation ===")
    for model, name in [
        (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), "RandomForest"),
        (GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42), "GradientBoosting"),
        (xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0), "XGBoost"),
    ]:
        evaluate(model, X_train, y_train.values, X_test, y_test.values, name)

    print(f"\n=== DONE — Total runtime: {time.time()-total_start:.1f}s ===")
    print(f"Extraction: {t1+t2:.1f}s | GA: {t_ga:.1f}s | Features: {len(selected)}")