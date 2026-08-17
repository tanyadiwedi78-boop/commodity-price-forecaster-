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


# ===============================================================
# DIRECTIONAL ACCURACY HELPER
# Measures whether predicted direction (up/down) matches actual.
# This is what the dashboard reports — optimize for it directly.
# ===============================================================

def directional_accuracy(y_true_prices, y_pred_prices):
    """Direction accuracy: does predicted day-to-day change match actual?"""
    y_true = np.asarray(y_true_prices, dtype=float)
    y_pred = np.asarray(y_pred_prices, dtype=float)
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    valid = actual_dir != 0
    if valid.sum() == 0:
        return 50.0
    return float(np.mean(actual_dir[valid] == pred_dir[valid]) * 100)


# ===============================================================
# LOAD DATA  (matches train.py: target = tomorrow's price,
#             target_return = tomorrow's % return)
# ===============================================================

def load_train_data(name, feat_cols):
    """Train portion only — don't touch test portion"""
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = df[df["commodity"] == name].sort_values("date").reset_index(drop=True)

    df["target"] = df["close_usd"].shift(-1)
    df["target_return"] = df["close_usd"].pct_change(1).shift(-1)
    df = df.dropna(subset=["target", "target_return"])

    train = df[df["date"] < TEST_START].copy()
    return train[feat_cols], train["target"], train["close_usd"]


def evaluate_baseline(X, y_prices, close_prices, n_splits=3):
    """Using default parameters present in settings.py (no tuning)"""
    params = LGBM_PARAMS.copy()
    if "objectives" in params:
        params["objective"] = params.pop("objectives")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores, da_scores = [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y_prices.iloc[train_idx]
        y_val = y_prices.iloc[val_idx]
        c_val = close_prices.iloc[val_idx].values

        # Compute target_return from target prices for training
        y_tr_ret = y_tr.pct_change().shift(-1).dropna()
        X_tr_aligned = X_tr.loc[y_tr_ret.index]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr_aligned, y_tr_ret)
        pred_ret = model.predict(X_val)
        pred_prices = c_val * (1 + pred_ret)

        rmse_scores.append(np.sqrt(mean_squared_error(y_val, pred_prices)))
        da_scores.append(directional_accuracy(y_val, pred_prices))

    return np.mean(rmse_scores), np.mean(da_scores)


# ===============================================================
# OBJECTIVE FUNCTION
# Optuna minimizes this. We use a blended metric:
#   score = -directional_accuracy + alpha * normalized_rmse
# This primarily rewards correct direction, with RMSE as tiebreaker.
# ===============================================================

def make_objective(X, y_prices, close_prices, n_splits=3):

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Pre-compute return targets (what the model actually learns)
    y_returns = y_prices.pct_change().shift(-1).dropna()
    X_aligned = X.loc[y_returns.index]
    close_aligned = close_prices.loc[y_returns.index]

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_da, fold_rmse = [], []
        for train_idx, val_idx in tscv.split(X_aligned):
            X_tr = X_aligned.iloc[train_idx]
            X_val = X_aligned.iloc[val_idx]
            y_tr = y_returns.iloc[train_idx]
            y_val_prices = y_prices.iloc[val_idx]
            c_val = close_aligned.iloc[val_idx].values

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            pred_ret = model.predict(X_val)
            pred_prices = c_val * (1 + pred_ret)

            da = directional_accuracy(y_val_prices, pred_prices)
            rmse = np.sqrt(mean_squared_error(y_val_prices, pred_prices))
            fold_da.append(da)
            fold_rmse.append(rmse)

        avg_da = np.mean(fold_da)
        avg_rmse = np.mean(fold_rmse)

        # Normalize RMSE by mean price to make it scale-independent
        mean_price = close_aligned.mean()
        norm_rmse = avg_rmse / (mean_price + 1e-9)

        # Primary: maximize directional accuracy (negate for minimization)
        # Secondary: minimize RMSE (weight=0.3 as tiebreaker)
        trial.set_user_attr("da", avg_da)
        trial.set_user_attr("rmse", avg_rmse)
        return -avg_da + 0.3 * norm_rmse * 100  # scale RMSE to similar range

    return objective


# ===============================================================
# TUNE ONE COMMODITY
# ===============================================================

def tune_commodity(name, feat_cols, n_trials=50):
    print(f"\n  Tuning: {name} ({n_trials} trials)")

    X, y_prices, close_prices = load_train_data(name, feat_cols)
    if len(X) < 100:
        print(f"    Skipped {name} -- not enough training rows")
        return None

    baseline_rmse, baseline_da = evaluate_baseline(X, y_prices, close_prices)
    print(f"    Baseline RMSE: {baseline_rmse:.4f}  |  Dir.Acc: {baseline_da:.1f}%")

    objective = make_objective(X, y_prices, close_prices)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_da = study.best_trial.user_attrs.get("da", 0)
    best_rmse = study.best_trial.user_attrs.get("rmse", 0)
    da_improvement = best_da - baseline_da

    print(f"    Best  RMSE: {best_rmse:.4f}  |  Dir.Acc: {best_da:.1f}%  "
          f"(DA improvement: {da_improvement:+.1f}pp)")
    print(f"    Best params: {study.best_params}")

    # Save this commodity's full trial-by-trial history
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    safe_name = name.replace(" ", "_").lower()
    study.trials_dataframe().to_csv(
        f"{OPTUNA_DIR}optuna_history_{safe_name}.csv", index=False
    )

    return {
        "best_params": study.best_params,
        "baseline_rmse": round(float(baseline_rmse), 4),
        "baseline_da": round(float(baseline_da), 2),
        "best_rmse": round(float(best_rmse), 4),
        "best_da": round(float(best_da), 2),
        "da_improvement": round(float(da_improvement), 2),
    }


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def tune_all(n_trials=50):
    print("=" * 55)
    print("HYPERPARAMETER TUNING (Optuna)")
    print("Objective: maximize directional accuracy + RMSE tiebreaker")
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
                "baseline_da": result["baseline_da"],
                "best_rmse": result["best_rmse"],
                "best_da": result["best_da"],
                "da_improvement": result["da_improvement"],
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
    tune_all(n_trials=50)
