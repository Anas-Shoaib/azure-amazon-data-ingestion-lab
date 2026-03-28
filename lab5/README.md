# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

## Dataset
NASA C-MAPSS Turbofan Engine Degradation (FD001)
- 100 training engines, 100 test engines
- 21 sensor readings + 3 operational settings per cycle

## Pipeline
```
Raw C-MAPSS → Preprocess → tsfresh Extraction → Variance Filter → Correlation Filter → MI Filter → Genetic Algorithm (DEAP) → Model Training
```

## Results

### Feature Selection
| Step | Features | Time |
|------|----------|------|
| Raw tsfresh | 85 | 0.5s |
| After variance filter | 40 | - |
| After correlation filter | 28 | - |
| After MI filter | 28 | - |
| After GA | 15 | 50.9s |

### Model Performance
| Model | RMSE | NASA Score |
|-------|------|------------|
| RandomForest | 51.61 | 8798 |
| GradientBoosting | 51.51 | 9286 |
| XGBoost | 51.14 | 8644 |

**Best model: XGBoost — RMSE=51.14**  
**Total pipeline runtime: 52.2s**

## How to Run
```bash
pip install tsfresh deap pandas numpy scikit-learn xgboost lightgbm pyarrow
cd lab5/src
python pipeline.py
```