import pandas as pd
import numpy as np
import joblib, os, sys, logging
import io

sys.path.append(".")
sys.path.append("..")

from config.settings import (
    COMMODITIES, PROCESSED_PATH, MODELS_PATH,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD,
    EMA_PERIOD, ATR_PERIOD, ROLLING_WINDOWS,
    LOG_PATH
)
from src.data_ingestion import load_raw



os.makedirs(LOG_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"{LOG_PATH}features.log",
            encoding='utf-8'
        ),
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
        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg)


log = logging.getLogger(__name__)


#================================================================
# TECHNICAL INDICATORS
#================================================================

def add_RSI(df, period=RSI_PERIOD):
    """ RSI -- Relative Strength Index """
    delta = df["close_usd"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_g / (avg_l + 1e-9)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """ MACD -- Momentum Indicator """
    ema_fast = df["close_usd"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close_usd"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_sig"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_his"] = df["macd"] - df["macd_sig"]
    return df


def add_bollinger(df, period=BOLLINGER_PERIOD, std=BOLLINGER_STD):
    """ Bollinger Bands = Volatility indicator """
    ma = df["close_usd"].rolling(period).mean()
    sd = df["close_usd"].rolling(period).std()
    df["bb_upper"] = ma + std * sd
    df["bb_lower"] = ma - std * sd
    df["bb_mid"] = ma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma + 1e-9)
    df["bb_pct"] = (df["close_usd"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    return df


def add_ema(df, periods=EMA_PERIOD):
    """ EXPONENTIAL MOVING AVERAGES """
    for p in periods:
        df[f"ema_{p}"] = df["close_usd"].ewm(span=p, adjust=False).mean()

    # EMA crossovers (computed once, after all EMAs exist)
    if 9 in periods and 21 in periods:
        df["ema_cross_9_21"] = df["ema_9"] - df["ema_21"]

    if 50 in periods and 200 in periods:
        df["ema_cross_50_200"] = df["ema_50"] - df["ema_200"]

    return df


def add_atr(df, period=ATR_PERIOD):
    """ ATR - Average true range (volatility) """
    hl = df["high_usd"] - df["low_usd"]
    hc = abs(df["high_usd"] - df["close_usd"].shift())
    lc = abs(df["low_usd"] - df["close_usd"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.rolling(period).mean()
    df["volatility_pct"] = df[f"atr_{period}"] / (df["close_usd"] + 1e-9) * 100
    return df


#===========================================================
# PRICE AND VOLUME FEATURES
#===========================================================
def add_lag_features(df):
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df[f"close_lag{lag}"] = df["close_usd"].shift(lag)
        df[f"return_lag{lag}"] = df["daily_return"].shift(lag)
    return df


def add_rolling_features(df, windows=ROLLING_WINDOWS):
    """ Rolling mean/std/min/max over configured windows """
    for w in windows:
        df[f"roll_mean_{w}"] = df["close_usd"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["close_usd"].rolling(w).std()
        df[f"roll_min_{w}"] = df["close_usd"].rolling(w).min()
        df[f"roll_max_{w}"] = df["close_usd"].rolling(w).max()
    return df


def add_return_features(df):
    df["return_1d"] = df["close_usd"].pct_change(1)
    df["return_5d"] = df["close_usd"].pct_change(5)
    df["return_30d"] = df["close_usd"].pct_change(30)
    df["return_90d"] = df["close_usd"].pct_change(90)
    df["return_252d"] = df["close_usd"].pct_change(252)

    # momentum
    df["momentum_14"] = df["close_usd"] - df["close_usd"].shift(14)
    df["momentum_30"] = df["close_usd"] - df["close_usd"].shift(30)
    return df


#==============================================================
# TIME / SEASONALITY FEATURES
#==============================================================
def add_time_features(df):
    """seasonality - month, quarter etc """
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end
    df["is_quarter_end"] = df["date"].dt.is_quarter_end

    # cyclic encoding (sin/cos) better than one-hot encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Year normalized
    df["year_norm"] = (df["year"] - 2010) / (2030 - 2010)

    return df


# ════════════════════════════════════════════════
# CROSS-COMMODITY FEATURES
# ════════════════════════════════════════════════

def add_cross_commodity_features(df_all):
    """
    Cross-commodity ratios:
    Gold/Silver ratio — precious metal relationship
    Oil/Gas ratio     — energy relationship
    Copper/Gold ratio — economic cycle indicator
    """
    log_print("  Adding cross-commodity features...")

    # Pivot to wide format
    wide = df_all.pivot_table(
        index="date",
        columns="commodity",
        values="close_usd"
    ).reset_index()

    # Gold-Silver ratio
    if "Gold" in wide.columns and "Silver" in wide.columns:
        wide["gold_silver_ratio"] = wide["Gold"] / (wide["Silver"] + 1e-9)

    # Oil-Gas ratio
    if "Crude oil" in wide.columns and "Natural Gas" in wide.columns:
        wide["oil_gas_ratio"] = wide["Crude oil"] / (wide["Natural Gas"] + 1e-9)

    # Copper-Gold ratio (economic indicator)
    if "Copper" in wide.columns and "Gold" in wide.columns:
        wide["copper_gold_ratio"] = wide["Copper"] / (wide["Gold"] + 1e-9)

    # Lag the ratios
    ratio_cols = [c for c in wide.columns
                if "ratio" in c or c in list(COMMODITIES.keys())]
    for col in ratio_cols:
        if col in wide.columns:
            wide[f"{col}_lag1"] = wide[col].shift(1)
            wide[f"{col}_lag7"] = wide[col].shift(7)
            wide[f"{col}_lag30"] = wide[col].shift(30)

    # Merge back
    merge_cols = [c for c in wide.columns if c != "date"
                and c not in list(COMMODITIES.keys())]
    df_all = df_all.merge(wide[["date"] + merge_cols],
                        on="date", how="left")

    log_print(f"  Cross-commodity features added: {len(merge_cols)}")
    return df_all


# ════════════════════════════════════════════════
# MONTHLY AGGREGATION
# ════════════════════════════════════════════════

def create_monthly_df(df):
    """
    Daily → Monthly aggregation
    For monthly price forecast
    """
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = (df.groupby(["commodity", "year_month"])
            .agg(
                date=("date", "last"),
                open_usd=("open_usd", "first"),
                high_usd=("high_usd", "max"),
                low_usd=("low_usd", "min"),
                close_usd=("close_usd", "last"),
                close_mean=("close_usd", "mean"),
                volume=("volume", "sum"),
                daily_ret_mean=("daily_return", "mean"),
                daily_ret_std=("daily_return", "std"),
                volatility=("volatility_pct", "mean"),
            ).reset_index())

    monthly["monthly_return"] = monthly.groupby("commodity")["close_usd"].pct_change()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.month
    monthly["quarter"] = pd.to_datetime(monthly["date"]).dt.quarter
    monthly["year"] = pd.to_datetime(monthly["date"]).dt.year
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)
    monthly["year_norm"] = (monthly["year"] - 2010) / 20

    # Monthly lags
    g = monthly.groupby("commodity")
    for lag in [1, 2, 3, 6, 12]:
        monthly[f"close_lag{lag}m"] = g["close_usd"].shift(lag)
        monthly[f"ret_lag{lag}m"] = g["monthly_return"].shift(lag)

    monthly["roll3m_mean"] = g["close_usd"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    monthly["roll6m_mean"] = g["close_usd"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    monthly["roll12m_mean"] = g["close_usd"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean())

    return monthly


# ════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════

def build_features():
    log_print("=" * 55)
    log_print("FEATURE ENGINEERING")
    log.info("=" * 55)

    # Load all raw data
    df_all = load_raw()
    if df_all is None or df_all.empty:
        log_print("No raw data! Run data_ingestion.py first")
        sys.exit(1)

    log_print(f"Loaded raw: {df_all.shape}")

    all_dfs = []

    for name in COMMODITIES:
        log.info(f"\n  Processing {COMMODITIES[name]['icon']} {name}...")
        df = df_all[df_all["commodity"] == name].copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Add all features
        df = add_RSI(df)
        df = add_macd(df)
        df = add_bollinger(df)
        df = add_ema(df)
        df = add_atr(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_return_features(df)
        df = add_time_features(df)

        log_print(f"    Features so far: {df.shape[1]}")
        all_dfs.append(df)

    # Combine
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Cross-commodity features
    df_combined = add_cross_commodity_features(df_combined)

    # Monthly aggregation
    df_monthly = create_monthly_df(df_combined)

    # Feature columns (exclude non-features)
    exclude = [
        "date", "commodity", "ticker", "unit", "sector",
        "year_month", "open_usd", "high_usd", "low_usd",
        "close_usd", "volume",  # raw OHLCV
    ]
    feat_cols = [c for c in df_combined.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(df_combined[c])]

    monthly_exclude = exclude + ["close_mean"]
    monthly_feat_cols = [c for c in df_monthly.columns
                        if c not in monthly_exclude
                        and pd.api.types.is_numeric_dtype(df_monthly[c])]

    # Drop rows with too many NaN (first ~200 days due to indicators)
    df_combined = df_combined.dropna(
        subset=["close_lag1", f"rsi_{RSI_PERIOD}"]
    ).reset_index(drop=True)
    df_monthly = df_monthly.dropna(
        subset=["close_lag1m"]
    ).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    df_combined.to_csv(PROCESSED_PATH, index=False)
    df_monthly.to_csv(
        PROCESSED_PATH.replace(".csv", "_monthly.csv"), index=False)
    joblib.dump(feat_cols,
                f"{MODELS_PATH}daily_feature_cols.pkl")
    joblib.dump(monthly_feat_cols,
                f"{MODELS_PATH}monthly_feature_cols.pkl")

    log_print(f"\n{'='*55}")
    log_print("FEATURE SUMMARY:")
    log_print(f"{'='*55}")
    log_print(f"  Daily  dataset: {df_combined.shape}")
    log_print(f"  Monthly dataset:{df_monthly.shape}")
    log_print(f"  Daily features: {len(feat_cols)}")
    log_print(f"  Monthly features:{len(monthly_feat_cols)}")

    # Per-commodity stats
    log_print("\n  Per-commodity rows:")
    for name in COMMODITIES:
        d = df_combined[df_combined["commodity"] == name]
        m = df_monthly[df_monthly["commodity"] == name]
        icon = COMMODITIES[name]["icon"]
        log_print(f"  {icon} {name:15s} "
                f"Daily: {len(d):>5,} | Monthly: {len(m):>4}")

    return df_combined, df_monthly, feat_cols, monthly_feat_cols


if __name__ == "__main__":
    if sys.platform == "win32":

        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    df_daily, df_monthly, feat_cols, m_feat_cols = build_features()

    print(f"\nDaily features ({len(feat_cols)}):")
    cats = {
        "Price Lags":      [f for f in feat_cols if "lag" in f and "close" in f],
        "Technical":       [f for f in feat_cols if any(x in f for x in ["rsi", "macd", "bb_", "ema_", "atr"])],
        "Rolling":         [f for f in feat_cols if "roll" in f],
        "Returns":         [f for f in feat_cols if "return" in f or "momentum" in f],
        "Time/Season":     [f for f in feat_cols if any(x in f for x in ["month", "quarter", "year", "sin", "cos"])],
        "Cross-Commodity": [f for f in feat_cols if "ratio" in f],
    }
    for cat, feats in cats.items():
        print(f"  {cat:20s}: {len(feats):>3} features")