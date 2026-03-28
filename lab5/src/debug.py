import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_parquet('../outputs/ga_train.parquet')
X_test  = pd.read_parquet('../outputs/ga_test.parquet')
y_train = pd.read_csv('../outputs/labels_train.csv', index_col=0).squeeze()
y_test  = pd.read_csv('../outputs/labels_test.csv',  index_col=0).squeeze()

m = RandomForestRegressor(n_estimators=50, random_state=42)
m.fit(X_train, y_train.values)
preds = m.predict(X_test)

print('Preds sample:', preds[:10])
print('y_test sample:', y_test.values[:10])
print('X_test shape:', X_test.shape)
print('X_test sample:\n', X_test.head())
print('X_train sample:\n', X_train.head())