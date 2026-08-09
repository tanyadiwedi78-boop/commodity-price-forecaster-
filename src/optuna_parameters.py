import pandas as pd
import numpy as np
import joblib, os, sys, json
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(".")
sys.path.append("..")

from config.settings import COMMODITIES, PROCESSED_PATH, MODELS_PATH, TEST_START, LGBM_PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_DIR = f"{MODELS_PATH}optuna/"

#=======================================================
# LOAD DATA
#=======================================================

def load_train_data(name, feat_cols):
    """Train portion only -- don't touch test portion"""
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = df[df["commodity"] == name].sort_values("date").reset_index(drop=True)

    df["target"] = df["close_usd"].shift(-1)
    df = df.dropna(subset=["target"])

    train = df[df["date"] < TEST_START].copy()
    return train[feat_cols], train["target"]


def evaluate_baseline(X, y, n_splits=3):
    """Using default parameters present in settings.py (no tuning)"""
    params = LGBM_PARAMS.copy()
    if "objectives" in params:
        params["objective"] = params.pop("objectives")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)


#=================================================
# OBJECTIVE FUNCTION (what optuna tries to minimize)
#==================================================

def make_objective(X, y, n_splits=3):

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            fold_scores.append(rmse)

        return np.mean(fold_scores)

    return objective


#=========================================
# TUNE ONE COMMODITY
#=========================================
def tune_commodity(name, feat_cols, n_trials=30):
    print(f"\n  Tuning: {name} ({n_trials} trials)")

    X, y = load_train_data(name, feat_cols)
    if len(X) < 100:
        print(f"    Skipped {name} -- not enough training rows")
        return None

    baseline_rmse = evaluate_baseline(X, y)
    print(f"    Baseline RMSE (settings.py defaults): {baseline_rmse:.4f}")

    objective = make_objective(X, y)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    improvement = ((baseline_rmse - study.best_value) / baseline_rmse * 100) if baseline_rmse else 0
    print(f"    Best RMSE: {study.best_value:.4f}  (improvement: {improvement:+.2f}%)")
    print(f"    Best params: {study.best_params}")

    # Save this commodity's full trial-by-trial history -- one CSV per commodity
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    safe_name = name.replace(" ", "_").lower()
    study.trials_dataframe().to_csv(
        f"{OPTUNA_DIR}optuna_history_{safe_name}.csv", index=False
    )

    return {
        "best_params": study.best_params,
        "baseline_rmse": round(float(baseline_rmse), 4),
        "best_rmse": round(float(study.best_value), 4),
        "improvement_pct": round(float(improvement), 2),
    }


#====================================================
# MAIN PIPELINE
#====================================================

def tune_all(n_trials=30):
    print("=" * 55)
    print("HYPERPARAMETER TUNING (Optuna)")
    print("=" * 55)

    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")
    best_params_all = {}
    summary_rows = []

    for name in COMMODITIES:
        result = tune_commodity(name, feat_cols, n_trials=n_trials)
        if result:
            best_params_all[name] = result["best_params"]
            summary_rows.append({
                "commodity": name,
                "baseline_rmse": result["baseline_rmse"],
                "best_rmse": result["best_rmse"],
                "improvement_pct": result["improvement_pct"],
            })

    # Save all tuned params to one file
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    params_path = f"{OPTUNA_DIR}tuned_lgbm_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params_all, f, indent=2)

    # Save the baseline-vs-tuned comparison table across all commodities
    summary_path = f"{OPTUNA_DIR}optuna_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\n{'='*55}")
    print("TUNING SUMMARY")
    print(f"{'='*55}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved: {params_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {OPTUNA_DIR}optuna_history_<commodity>.csv (one per commodity)")
    return best_params_all


if __name__ == "__main__":
    tune_all(n_trials=30)