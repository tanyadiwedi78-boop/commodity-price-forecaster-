import pandas as pd
import numpy as np
import joblib, os, sys, json
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(".")
sys.path.append("..")

from config.settings import (
    COMMODITIES, MODELS_PATH, PROCESSED_PATH,
    FORCAST_DAYS, ENSEMBLE_WEIGHTS, SARIMA_ORDER, SARIMA_SEASONAL
)
from src.db import get_engine, init_tables
from src.data_ingestion import  load_raw , convert_to_inr
from src.features import (
    add_RSI, add_macd, add_bollinger, add_ema, add_atr,
    add_lag_features, add_rolling_features, add_return_features,
    add_time_features, add_cross_commodity_features
)


# ════════════════════════════════════════════════
# METRICS + WEIGHTS
# ════════════════════════════════════════════════

def load_metrics(name):
    
    safe_name = name.replace(" ", "_").lower()
    path = f"{MODELS_PATH}all_metrics.json"
    if not os.path.exists(path):
        return {"rmse": None, "mape": None}
    with open(path) as f:
        all_metrics = json.load(f)
    m = all_metrics.get(safe_name, {})
    return {"rmse": m.get("rmse"), "mape": m.get("mape")}


def load_ensemble_weights(name):
    
    safe_name = name.replace(" ", "_").lower()
    path = f"{MODELS_PATH}optuna/ensemble_weights_{safe_name}.json"
    if os.path.exists(path):
        with open(path) as f:
            w = json.load(f)
        return {
            "lgbm": w.get("lgbm", w.get("w_lgbm")),
            "sarima": w.get("sarima", w.get("w_sarima")),
        }
    return ENSEMBLE_WEIGHTS


def get_recommendation(current_price, predicted_price, mape):
    if mape is None or mape == 0:
        mape = 2.0  # sensible fallback if metrics aren't available yet
    pct_change = ((predicted_price - current_price) / current_price) * 100
    if pct_change > mape:
        signal = "BUY"
        reason = f"Predicted +{pct_change:.2f}% exceeds the model's typical error margin ({mape:.2f}%)"
    elif pct_change < -mape:
        signal = "SELL"
        reason = f"Predicted {pct_change:.2f}% falls below the model's typical error margin (-{mape:.2f}%)"
    else:
        signal = "HOLD"
        reason = f"Predicted {pct_change:+.2f}% is within the model's noise range (±{mape:.2f}%) -- not a confident signal"
    return signal, reason, round(pct_change, 2)


def load_models(name):
    safe_name = name.replace(" ", "_").lower()
    lgbm_model = joblib.load(f"{MODELS_PATH}{safe_name}_lgbm.pkl")
    sarima_model = joblib.load(f"{MODELS_PATH}{safe_name}_sarima.pkl")
    return lgbm_model, sarima_model


def refit_sarima_on_full_data(df):
    series = df.set_index("date")["close_usd"]
    model = SARIMAX(
        series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False)


# ════════════════════════════════════════════════
# FEATURE RECOMPUTATION (training wale exact functions reuse)
# ════════════════════════════════════════════════

def recompute_features_for_all(df_all):

    all_dfs = []
    for name in COMMODITIES:
        df = df_all[df_all["commodity"] == name].copy()
        df = df.sort_values("date").reset_index(drop=True)
        df = add_RSI(df)
        df = add_macd(df)
        df = add_bollinger(df)
        df = add_ema(df)
        df = add_atr(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_return_features(df)
        df = add_time_features(df)
        all_dfs.append(df)
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = add_cross_commodity_features(df_combined)
    return df_combined


# ════════════════════════════════════════════════
# MAIN RECURSIVE ENGINE
# ════════════════════════════════════════════════

def generate_all_forecasts_recursive(df_all, feat_cols):
    
    df_working = df_all.copy()

    sarima_paths, usdinr_latest, conv_types = {}, {}, {}
    last_closes, last_dates, lgbm_models = {}, {}, {}

    for name in COMMODITIES:
        dfc = df_all[df_all["commodity"] == name].sort_values("date")
        sarima_fitted = refit_sarima_on_full_data(dfc)
        sarima_paths[name] = np.asarray(sarima_fitted.forecast(steps=FORCAST_DAYS))
        usdinr_latest[name] = float(dfc["usdinr"].iloc[-1]) if "usdinr" in dfc.columns else 83.5
        conv_types[name] = COMMODITIES[name].get("conversion")
        last_closes[name] = float(dfc["close_usd"].iloc[-1])
        last_dates[name] = dfc["date"].iloc[-1]
        lgbm_models[name], _ = load_models(name)

    all_rows = []

    for day in range(FORCAST_DAYS):  # day=0 -> horizon_days=1, day=1 -> horizon_days=2, ...
        df_feat = recompute_features_for_all(df_working)
        new_rows = []

        for name in COMMODITIES:
            latest = df_feat[df_feat["commodity"] == name].iloc[-1]
            X = latest[feat_cols].to_frame().T.astype(float)
            lgbm_return = float(lgbm_models[name].predict(X)[0])

            prev_close = (last_closes[name] if day == 0
                        else df_working[df_working["commodity"] == name]["close_usd"].iloc[-1])
            lgbm_price = prev_close * (1 + lgbm_return)
            sarima_price = sarima_paths[name][day]

            w = load_ensemble_weights(name)
            predicted = w["lgbm"] * lgbm_price + w["sarima"] * sarima_price

            metrics = load_metrics(name)
            rmse = metrics["rmse"] or 0
            mape = metrics["mape"]

            band_width_usd = rmse * ((day + 1) ** 0.5)
            ci_low_usd = predicted - band_width_usd
            ci_high_usd = predicted + band_width_usd

            forecast_date = (last_dates[name] + timedelta(days=day + 1)).date()

            def to_inr(price_usd, name=name):
                if not conv_types[name]:
                    return price_usd
                return convert_to_inr(price_usd, conv_types[name], usdinr_latest[name])

            signal, reason, pct_change = get_recommendation(prev_close, predicted, mape)

            all_rows.append({
                "commodity": name,
                "forecast_date": forecast_date,
                "predicted_price": round(to_inr(predicted), 2),
                "predicted_price_lgbm": round(to_inr(lgbm_price), 2),
                "predicted_price_sarima": round(to_inr(sarima_price), 2),
                "ci_low": round(to_inr(ci_low_usd), 2),
                "ci_high": round(to_inr(ci_high_usd), 2),
                "horizon_days": day + 1,
                "model_version": "v2_recursive",
                "recommendation": signal,
                "reasoning": reason,
                "pct_change": pct_change,
            })

            new_rows.append({
                "date": pd.Timestamp(forecast_date),
                "commodity": name,
                "open_usd": predicted, "high_usd": predicted,
                "low_usd": predicted, "close_usd": predicted,
                "volume": latest.get("volume", 0),
                "daily_return": pct_change / 100,
                "usdinr": usdinr_latest[name],
            })

        df_working = pd.concat([df_working, pd.DataFrame(new_rows)], ignore_index=True)

    return pd.DataFrame(all_rows)


def save_all_forecasts():
    init_tables()
    engine = get_engine()

    df_all = load_raw()
    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")

    final_df = generate_all_forecasts_recursive(df_all, feat_cols)
    final_df.to_sql("forecasts", engine, if_exists="append", index=False)
    print(f"\nSaved {len(final_df)} forecast rows to PostgreSQL")


if __name__ == "__main__":
    print("=" * 55)
    print("GENERATING & SAVING FORECASTS (Recursive Multi-Day)")
    print("=" * 55)
    save_all_forecasts()