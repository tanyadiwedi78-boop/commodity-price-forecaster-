"""
src/train_monthly.py
Monthly Forecasting Model -- LightGBM + SARIMA on monthly data.

Why this exists separately from train.py:
Daily data lets us forecast ~30 days ahead reliably, but pushing a DAILY
model out to 3 years compounds errors badly (see README/roadmap notes).
Monthly data is coarser -- each "step" is a month instead of a day -- so
a model trained on it can reliably forecast much further out (FORCAST_MONTHS
= 36, i.e. 3 years), the same way the reference project used quarterly
data to forecast 8 quarters (2 years) of GDP.

Run (after feature_engineering.py has produced merged_features_monthly.csv):
    python src/train_monthly.py
"""

import pandas as pd
import numpy as np
import joblib, os, sys, logging, json
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

sys.path.append(".")
sys.path.append("..")

from config.settings import (
    COMMODITIES, PROCESSED_PATH, MODELS_PATH,
    TEST_START, TEST_END, LGBM_PARAMS,
    SARIMA_ORDER, SARIMA_SEASONAL, ENSEMBLE_WEIGHTS,
    FORCAST_MONTHS, LOG_PATH
)

os.makedirs(LOG_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_PATH}train_monthly.log", encoding='utf-8'),
        logging.StreamHandler(stream=open(os.devnull, 'w', encoding='utf-8', errors='replace'))
    ]
)


def log_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


MONTHLY_PATH = PROCESSED_PATH.replace(".csv", "_monthly.csv")


# ════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(MONTHLY_PATH, parse_dates=["date"])
    feat_cols = joblib.load(f"{MODELS_PATH}monthly_feature_cols.pkl")
    return df, feat_cols


def make_target(df, horizon=1):
    """
    Same fix as train.py: LightGBM predicts % return (not raw price),
    since tree-based models can't extrapolate beyond the price range they
    trained on. This matters just as much for monthly data -- maybe more,
    since fewer data points means fewer examples of any given price level.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["target"] = df["close_usd"].shift(-horizon)
    df["target_return"] = df["close_usd"].pct_change(horizon).shift(-horizon)
    return df


def split_train_test(df):
    train = df[df["date"] < TEST_START].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    return train, test


# ════════════════════════════════════════════════
# TRAIN MODELS
# ════════════════════════════════════════════════

def train_lgbm(train, test, feat_cols):
    X_train, y_train = train[feat_cols], train["target_return"]
    X_test = test[feat_cols]

    params = LGBM_PARAMS.copy()
    if "objectives" in params:
        params["objective"] = params.pop("objectives")

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)
    price_preds = test["close_usd"].values * (1 + predicted_returns)
    return model, price_preds


def train_sarima(train, test):
    """Monthly SARIMA -- seasonal_order period should reflect ANNUAL
    seasonality in monthly data (12 months per cycle), same config as
    settings.py already defines."""
    series = train.set_index("date")["close_usd"]
    model = SARIMAX(
        series,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    preds = fitted.forecast(steps=len(test))
    return fitted, preds.values


# ════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════

def evaluate(y_true, y_pred, name):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        log_print(f"    [{name:9s}] No valid rows to evaluate")
        return None, None, None, None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    valid = actual_dir != 0
    dir_acc = (np.mean(actual_dir[valid] == pred_dir[valid]) * 100) if valid.sum() > 0 else 0

    log_print(f"    [{name:9s}] RMSE: {rmse:8.3f} | MAE: {mae:8.3f} | "
            f"MAPE: {mape:5.2f}% | Dir.Acc: {dir_acc:5.1f}%")
    return rmse, mae, mape, dir_acc


# ════════════════════════════════════════════════
# TRAIN ONE COMMODITY
# ════════════════════════════════════════════════

def train_commodity_monthly(name, df_all, feat_cols, all_metrics):
    log_print(f"\n  Training (monthly): {name}")

    df = df_all[df_all["commodity"] == name].copy()
    df = make_target(df, horizon=1)
    df = df.dropna(subset=["target", "target_return"])

    train, test = split_train_test(df)
    if len(test) < 3 or len(train) < 12:
        log_print(f"    Skipped {name} -- not enough monthly data "
                   f"(train={len(train)}, test={len(test)})")
        return None

    lgbm_model, lgbm_preds = train_lgbm(train, test, feat_cols)
    sarima_model, sarima_preds = train_sarima(train, test)

    w = ENSEMBLE_WEIGHTS
    ensemble_preds = (w["lgbm"] * lgbm_preds) + (w["sarima"] * sarima_preds)

    y_true = test["target"].values
    evaluate(y_true, lgbm_preds, "LightGBM")
    evaluate(y_true, sarima_preds, "SARIMA")
    ensemble_rmse, ensemble_mae, ensemble_mape, ensemble_dir_acc = evaluate(y_true, ensemble_preds, "Ensemble")

    safe_name = name.replace(" ", "_").lower()
    if ensemble_rmse is not None:
        all_metrics[safe_name] = {
            "commodity": name,
            "rmse": round(float(ensemble_rmse), 4),
            "mae": round(float(ensemble_mae), 4),
            "mape": round(float(ensemble_mape), 4),
            "directional_accuracy": round(float(ensemble_dir_acc), 1),
            "w_lgbm": w["lgbm"],
            "w_sarima": w["sarima"],
        }

    joblib.dump(lgbm_model, f"{MODELS_PATH}{safe_name}_lgbm_monthly.pkl")
    joblib.dump(sarima_model, f"{MODELS_PATH}{safe_name}_sarima_monthly.pkl")
    log_print(f"    Saved: {safe_name}_lgbm_monthly.pkl, {safe_name}_sarima_monthly.pkl")

    return {"lgbm": lgbm_model, "sarima": sarima_model}


# ════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════

def train_all_monthly():
    log_print("=" * 55)
    log_print("MONTHLY MODEL TRAINING (long-horizon forecasts)")
    log_print("=" * 55)

    if not os.path.exists(MONTHLY_PATH):
        log_print(f"ERROR: {MONTHLY_PATH} not found.")
        log_print("Run feature_engineering.py first -- it builds this file.")
        sys.exit(1)

    df_all, feat_cols = load_data()
    log_print(f"Loaded monthly data: {df_all.shape}")
    log_print(f"Using {len(feat_cols)} monthly features")
    log_print(f"Max forecast horizon (settings.py): {FORCAST_MONTHS} months "
               f"(~{FORCAST_MONTHS/12:.1f} years)")

    os.makedirs(MODELS_PATH, exist_ok=True)
    results = {}
    all_metrics = {}

    for name in COMMODITIES:
        result = train_commodity_monthly(name, df_all, feat_cols, all_metrics)
        if result:
            results[name] = result

    with open(f"{MODELS_PATH}all_metrics_monthly.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log_print(f"\nSaved: {MODELS_PATH}all_metrics_monthly.json")

    log_print(f"\n{'='*55}")
    log_print(f"MONTHLY TRAINING COMPLETE: {len(results)}/{len(COMMODITIES)} commodities trained")
    log_print(f"{'='*55}")
    return results


if __name__ == "__main__":
    train_all_monthly()