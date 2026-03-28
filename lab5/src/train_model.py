import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1).sum()

def evaluate(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, 125)
    print("  preds sample:", preds[:5])
    print("  y_test sample:", y_test[:5])
    print(f"[{name}] RMSE={rmse(y_test,preds):.2f}  NASA Score={nasa_score(y_test,preds):.0f}")

if __name__ == "__main__":
    X_train = pd.read_parquet("../outputs/ga_train.parquet")
    X_test  = pd.read_parquet("../outputs/ga_test.parquet")
    y_train = pd.read_csv("../outputs/labels_train.csv", index_col=0).squeeze()
    y_test  = pd.read_csv("../outputs/labels_test.csv",  index_col=0).squeeze()

    evaluate(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
             X_train, y_train.values, X_test, y_test.values, "RandomForest")

    evaluate(GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
             X_train, y_train.values, X_test, y_test.values, "GradientBoosting")

    evaluate(xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1, verbosity=0),
             X_train, y_train.values, X_test, y_test.values, "XGBoost")