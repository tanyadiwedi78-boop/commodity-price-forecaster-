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
    LOG_PATH
)

os.makedirs(LOG_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_PATH}train.log", encoding='utf-8'),
        logging.StreamHandler(
            stream=open(os.devnull, 'w', encoding='utf-8', errors='replace')
        )
    ]
)


def log_print(msg):
    """Safe print for Windows terminal"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


log = logging.getLogger(__name__)


# ════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════

def load_data():
    """Load processed daily data + saved feature column list"""
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")
    return df, feat_cols


# ════════════════════════════════════════════════
# STEP 2: CREATE TARGET (what we're predicting)
# ════════════════════════════════════════════════

def make_target(df, horizon=1):
    
    df = df.sort_values("date").reset_index(drop=True)
    df["target"] = df["close_usd"].shift(-horizon)
    df["target_return"] = df["close_usd"].pct_change(horizon).shift(-horizon)
    return df


# ════════════════════════════════════════════════
# STEP 3: TIME-BASED SPLIT (no random split for time series!)
# ════════════════════════════════════════════════

def split_train_test(df):
    train = df[df["date"] < TEST_START].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    return train, test


# ════════════════════════════════════════════════
# STEP 4: TRAIN MODELS
# ════════════════════════════════════════════════

def load_tuned_params(name):
    """
    If tune_hyperparameters.py has been run, use its best params for this
    commodity. Otherwise fall back to the defaults in settings.py.
    """
    path = f"{MODELS_PATH}optuna/tuned_lgbm_params.json"
    if os.path.exists(path):
        with open(path) as f:
            tuned = json.load(f)
        if name in tuned:
            return tuned[name]
    return None


def train_lgbm(train, test, feat_cols, name):
    X_train, y_train = train[feat_cols], train["target_return"]
    X_test = test[feat_cols]

    tuned = load_tuned_params(name)
    if tuned:
        log_print(f"    Using tuned hyperparameters for {name}")
        params = {"objective": "regression", "metric": "rmse",
                  "verbose": -1, "random_state": 42, **tuned}
    else:
        params = LGBM_PARAMS.copy()
        if "objectives" in params:
            params["objective"] = params.pop("objectives")

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)
    price_preds = test["close_usd"].values * (1 + predicted_returns)

    return model, price_preds, predicted_returns  


def train_sarima(train, test):
    """Univariate SARIMA on close price -- captures trend/seasonality"""
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
# STEP 5: EVALUATE
# ════════════════════════════════════════════════

def evaluate(y_true, y_pred, name):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

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

def evaluate_direction_from_returns(y_true_returns, y_pred_returns, name):
    
    y_true_returns = np.asarray(y_true_returns, dtype=float)
    y_pred_returns = np.asarray(y_pred_returns, dtype=float)
    mask = ~np.isnan(y_true_returns) & ~np.isnan(y_pred_returns)
    y_true_returns, y_pred_returns = y_true_returns[mask], y_pred_returns[mask]

    actual_sign = np.sign(y_true_returns)
    pred_sign = np.sign(y_pred_returns)
    valid = actual_sign != 0
    dir_acc = (np.mean(actual_sign[valid] == pred_sign[valid]) * 100) if valid.sum() > 0 else 0

    log_print(f"    [{name:9s}] Return-based Dir.Acc: {dir_acc:5.1f}%")
    return dir_acc


# ════════════════════════════════════════════════
# STEP 6: TRAIN ONE COMMODITY (LGBM + SARIMA + ENSEMBLE)
# ════════════════════════════════════════════════

def train_commodity(name, df_all, feat_cols, all_metrics):
    log_print(f"\n  Training: {name}")

    df = df_all[df_all["commodity"] == name].copy()
    df = make_target(df, horizon=1)
    df = df.dropna(subset=["target", "target_return"])  # last row + first row (no prior price) both drop

    train, test = split_train_test(df)
    if len(test) == 0 or len(train) == 0:
        log_print(f"    Skipped {name} -- no data in train/test window")
        return None

    lgbm_model, lgbm_preds , lgbm_predicted_returns = train_lgbm(train, test, feat_cols, name)
    sarima_model, sarima_preds = train_sarima(train, test)

    # Weighted ensemble (weights from settings.py)
    w = ENSEMBLE_WEIGHTS
    ensemble_preds = (w["lgbm"] * lgbm_preds) + (w["sarima"] * sarima_preds)

    y_true = test["target"].values
    evaluate(y_true, lgbm_preds, "LightGBM")
    evaluate(y_true, sarima_preds, "SARIMA")
    ensemble_rmse, ensemble_mae, ensemble_mape, ensemble_dir_acc = evaluate(y_true, ensemble_preds, "Ensemble")
    lgbm_return_dir_acc = evaluate_direction_from_returns(
        test["target_return"].values, lgbm_predicted_returns, "LightGBM"
    )
    
    safe_name = name.replace(" ", "_").lower()
    all_metrics[safe_name] = {
        "commodity": name,
        "rmse": round(float(ensemble_rmse), 4),
        "mae": round(float(ensemble_mae), 4),
        "mape": round(float(ensemble_mape), 4),
        "directional_accuracy": round(float(ensemble_dir_acc), 1),
        "directional_accuracy_return_based": round(float(lgbm_return_dir_acc), 1),
        "w_lgbm": ENSEMBLE_WEIGHTS["lgbm"],
        "w_sarima": ENSEMBLE_WEIGHTS["sarima"],
    }

    # Save actual vs predicted for the backtest chart (plot_backtest.py reads this)
    backtest_df = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_true,
        "predicted": ensemble_preds,
    })
    backtest_df.to_csv(f"{MODELS_PATH}backtest_{safe_name}.csv", index=False)

    # Save models
    safe_name = name.replace(" ", "_").lower()
    joblib.dump(lgbm_model, f"{MODELS_PATH}{safe_name}_lgbm.pkl")
    joblib.dump(sarima_model, f"{MODELS_PATH}{safe_name}_sarima.pkl")
    log_print(f"    Saved: {safe_name}_lgbm.pkl, {safe_name}_sarima.pkl")

    return {"lgbm": lgbm_model, "sarima": sarima_model}


# ════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════

def train_all():
    log_print("=" * 55)
    log_print("MODEL TRAINING")
    log_print("=" * 55)

    df_all, feat_cols = load_data()
    log_print(f"Loaded processed data: {df_all.shape}")
    log_print(f"Using {len(feat_cols)} features")
    log_print(f"Train/Test window: before {TEST_START}  |  test {TEST_START} -> {TEST_END}")

    os.makedirs(MODELS_PATH, exist_ok=True)
    results = {}
    all_metrics = {}

    for name in COMMODITIES:
        result = train_commodity(name, df_all, feat_cols, all_metrics)
        if result:
            results[name] = result

    # Save ALL commodities' metrics in ONE combined file
    log_print(f"\nCommodities with metrics collected: {list(all_metrics.keys())}")
    with open(f"{MODELS_PATH}all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log_print(f"Saved combined metrics: {MODELS_PATH}all_metrics.json")

    log_print(f"\n{'='*55}")
    log_print(f"TRAINING COMPLETE: {len(results)}/{len(COMMODITIES)} commodities trained")
    log_print(f"{'='*55}")
    return results


if __name__ == "__main__":
    train_all()