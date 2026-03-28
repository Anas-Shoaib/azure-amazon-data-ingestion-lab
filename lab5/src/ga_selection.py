import time
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def run_ga(X_train, y_train, pop=30, gen=15):
    n = X_train.shape[1]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        selected = [i for i, b in enumerate(ind) if b == 1]
        if len(selected) == 0:
            return (9999.0,)
        X_sel = X_train.iloc[:, selected]
        model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42, n_jobs=2)
        scores = cross_val_score(model, X_sel, y_train, cv=3, scoring="neg_root_mean_squared_error")
        return (-scores.mean(),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate",   tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    t0 = time.time()
    pop_ = toolbox.population(n=pop)
    hof  = tools.HallOfFame(1)
    algorithms.eaSimple(pop_, toolbox, cxpb=0.7, mutpb=0.2, ngen=gen, halloffame=hof, verbose=False)

    best = hof[0]
    selected_idx  = [i for i, b in enumerate(best) if b == 1]
    selected_cols = X_train.columns[selected_idx].tolist()
    elapsed = time.time() - t0
    print(f"GA done in {elapsed:.1f}s — {len(selected_cols)} features selected")
    print("Selected:", selected_cols)
    return selected_cols, elapsed

if __name__ == "__main__":
    X_train = pd.read_parquet("../outputs/filtered_train.parquet")
    X_test  = pd.read_parquet("../outputs/filtered_test.parquet")
    y_train = pd.read_csv("../outputs/labels_train.csv", index_col=0).squeeze()

    selected = run_ga(X_train, y_train)
    pd.Series(selected).to_csv("../outputs/selected_features.csv", index=False)
    X_train[selected].to_parquet("../outputs/ga_train.parquet")
    X_test[selected].to_parquet("../outputs/ga_test.parquet")
    print("Saved.")