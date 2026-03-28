import pandas as pd
import numpy as np

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1","sensor_2","sensor_3","sensor_4","sensor_5",
    "sensor_6","sensor_7","sensor_8","sensor_9","sensor_10",
    "sensor_11","sensor_12","sensor_13","sensor_14","sensor_15",
    "sensor_16","sensor_17","sensor_18","sensor_19","sensor_20","sensor_21"
]

CONSTANT_SENSORS = ["sensor_1","sensor_5","sensor_6","sensor_10","sensor_16","sensor_18","sensor_19"]

def load_cmapss(train_path, test_path, rul_path):
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COLUMNS)
    rul_df   = pd.read_csv(rul_path,   sep=r"\s+", header=None, names=["RUL"])

    max_cycles = train_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="engine_id")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop(columns=["max_cycle"], inplace=True)

    rul_df["engine_id"] = rul_df.index + 1
    test_max = test_df.groupby("engine_id")["cycle"].max().reset_index()
    test_max.columns = ["engine_id", "max_cycle"]
    test_df = test_df.merge(test_max, on="engine_id")
    test_df = test_df.merge(rul_df, on="engine_id")
    test_df["RUL"] = test_df["RUL"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df.drop(columns=["max_cycle"], inplace=True)

    return train_df, test_df

def preprocess(df):
    df = df.copy()
    df.drop(columns=[c for c in CONSTANT_SENSORS if c in df.columns], inplace=True)
    if "RUL" in df.columns:
        df["RUL"] = df["RUL"].clip(upper=125)
    return df