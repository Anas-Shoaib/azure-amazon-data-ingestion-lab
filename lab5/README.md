# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

## Overview
This lab builds a full feature engineering and model training pipeline for predicting the **Remaining Useful Life (RUL)** of aircraft engines using the NASA C-MAPSS Turbofan Engine Degradation dataset (FD001 subset).

The pipeline covers: data loading → preprocessing → tsfresh feature extraction → filter-based selection → genetic algorithm selection → regression model evaluation.

---

## Dataset
**NASA C-MAPSS FD001**
- 100 training engines, 100 test engines
- 21 sensor readings + 3 operational settings recorded per cycle
- Each engine runs from healthy state until failure
- Target: predict how many cycles remain before failure (RUL)

Downloaded from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ (item #6)

---

## Pipeline Architecture

```
NASA C-MAPSS raw text files
        ↓
load_data.py        — parse, compute RUL, drop constant sensors, clip RUL at 125
        ↓
extract_features.py — tsfresh feature extraction (mean, std, min, max, median per sensor)
        ↓
filter_selection.py — variance filter → correlation filter → mutual information filter
        ↓
ga_selection.py     — DEAP genetic algorithm evolves optimal feature subset
        ↓
train_model.py      — RandomForest, GradientBoosting, XGBoost evaluation
        ↓
pipeline.py         — orchestrates all steps end-to-end
```

---

## Results

### Feature Reduction
| Step | Features Remaining | Time |
|------|--------------------|------|
| Raw tsfresh output | 85 | 0.5s |
| After variance filter (threshold=0.01) | 40 | — |
| After correlation filter (threshold=0.99) | 28 | — |
| After mutual information filter (top 28) | 28 | — |
| After genetic algorithm | **15** | 50.9s |

### Final Selected Features
```
sensor_9__mean, sensor_4__minimum, sensor_2__mean, sensor_3__minimum,
sensor_12__maximum, sensor_14__maximum, sensor_11__maximum, sensor_9__maximum,
sensor_2__minimum, sensor_17__mean, sensor_17__minimum, sensor_7__maximum,
sensor_9__standard_deviation, sensor_12__minimum, sensor_17__maximum
```

### Model Performance
| Model | RMSE | NASA Score |
|-------|------|------------|
| RandomForest | 51.61 | 8798 |
| GradientBoosting | 51.51 | 9286 |
| **XGBoost** | **51.14** | **8644** |

**Best model: XGBoost — RMSE=51.14, NASA Score=8644**  
**Total pipeline runtime: 52.2s**  
**Feature extraction time: 0.5s**

---

## Difficulties Encountered and How They Were Solved

### 1. Windows multiline terminal commands don't work
**Problem:** Trying to run multiline `python -c "..."` commands in the Windows terminal caused each line to be interpreted as a separate command, producing errors like `'from' is not recognized`.

**Fix:** Created small `.py` test files instead and ran those directly.

---

### 2. tsfresh `sum_values` feature caused all-zero predictions
**Problem:** The initial feature set used `MinimalFCParameters` which includes `sum_values`. This feature sums all sensor readings across the entire time series — so training engines (which ran until failure, hundreds of cycles) had much larger values than test engines (which were cut short). The model learned that high `sum_values` = high RUL, but test engines all had low sums, so it predicted 0 for everything.

**Fix:** Replaced `MinimalFCParameters` with a custom feature set using only statistics that are scale-independent: `mean`, `standard_deviation`, `maximum`, `minimum`, `median`.

---

### 3. Per-engine normalization destroyed all feature variance
**Problem:** After fixing the feature type, all three models still predicted the same identical RMSE (84.55). The cause was per-engine min-max normalization in preprocessing — normalizing each sensor to [0,1] within each engine made every engine look identical, so features like `mean` and `max` became the same for all engines.

**Fix:** Removed per-engine normalization entirely. Tree-based models (RF, XGBoost, GBM) don't require normalization, so this was unnecessary anyway.

---

### 4. Labels were all zero — model learned to always predict 0
**Problem:** Even after fixing normalization, predictions were still all zero. The training labels (`y_train`) were all 0 because we extracted features from each engine's full time series and took the RUL at the last cycle — which is always 0 (the engine just failed).

**Fix:** Changed `prepare_for_tsfresh()` to use only the **first 70% of each engine's cycles**, so the labels represent meaningful mid-life RUL values rather than end-of-life zeros.

---

### 5. `groupby().apply()` dropped the `engine_id` column on Windows/pandas
**Problem:** Using `df.groupby("engine_id").apply(lambda g: g.iloc[:...]).reset_index()` moved `engine_id` into the index, causing a `KeyError: ['engine_id'] not in index` when trying to select columns afterward.

**Fix:** Used `group_keys=False` in the groupby call, then rewrote it as an explicit loop to avoid the index issue entirely.

---

### 6. GA penalty too aggressive — selected only 1 feature
**Problem:** The genetic algorithm's fitness function penalized feature count too heavily (`* 0.5` per feature), causing it to always converge to selecting just 1 feature regardless of predictive value.

**Fix:** Removed the penalty entirely and let the GA optimize purely on cross-validated RMSE. With only 17-28 features going in, overfitting from too many features wasn't a real concern.

---

### 7. Correlation filter was too aggressive — left only 11 features
**Problem:** With a correlation threshold of 0.95, the filter dropped too many features, leaving only 11. The GA had too little to work with.

**Fix:** Raised the threshold to 0.99 to only remove near-perfect duplicates, preserving 28 features for the GA.

---

### 8. Windows multiprocessing crash in pipeline.py
**Problem:** Running `pipeline.py` caused a recursive multiprocessing crash on Windows. tsfresh uses `n_jobs=4` internally, which spawns worker processes. On Windows, Python re-imports the main module in each worker, which re-triggered the pipeline code infinitely.

**Fix:** Two changes — set `n_jobs=1` in tsfresh to disable multiprocessing, and wrapped all pipeline code in `if __name__ == '__main__':` to prevent re-execution on import.

---

### 9. `elapsed` variable undefined in ga_selection.py
**Problem:** The `run_ga` function returned `selected_cols, elapsed` but `elapsed` was never assigned — `time.time() - t0` was only used inside a print statement.

**Fix:** Added `elapsed = time.time() - t0` before the print and return statements.

---

## How to Run

```bash
# Install dependencies
pip install tsfresh deap pandas numpy scikit-learn xgboost lightgbm pyarrow

# Navigate to src
cd lab5/src

# Run full pipeline
python pipeline.py
```

Individual steps can also be run separately:
```bash
python extract_features.py   # tsfresh extraction → outputs/
python filter_selection.py   # filter pipeline → outputs/
python ga_selection.py       # GA selection → outputs/
python train_model.py        # model evaluation
```

---

## File Structure
```
lab5/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── src/
│   ├── load_data.py          # data loading and preprocessing
│   ├── extract_features.py   # tsfresh feature extraction
│   ├── filter_selection.py   # variance, correlation, MI filters
│   ├── ga_selection.py       # DEAP genetic algorithm
│   ├── train_model.py        # model training and evaluation
│   └── pipeline.py           # end-to-end orchestration
└── outputs/
    ├── features_train.parquet
    ├── features_test.parquet
    ├── filtered_train.parquet
    ├── filtered_test.parquet
    ├── ga_train.parquet
    ├── ga_test.parquet
    ├── labels_train.csv
    ├── labels_test.csv
    └── selected_features.csv
```
