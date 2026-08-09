import yfinance as yf
import pandas as pd
import numpy as np
import os, sys, time, logging
from datetime import datetime
import io

sys.path.append(".")
sys.path.append("..")



from config.settings import (
    COMMODITIES, START_DATE, END_DATE,
    USDINR_TICKER, RAW_PATH, LOG_PATH,
    CURRENCY_SYMBOL
)

os.makedirs(LOG_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"{LOG_PATH}ingestion.log",
            encoding='utf-8'         
        ),
        logging.StreamHandler(
            stream=open(os.devnull, 'w')  
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


def fetch_usdinr_rate():
    log_print("\n[1/6] Fetching USD/INR rate...")

    cache = f"{RAW_PATH}usdinr_rate.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["date"])
        rate = df['usdinr'].iloc[-1]
        log_print(f"    Cache found! Latest rate: Rs.{rate:.2f}")
        return df

    try:
        df = yf.download(
            USDINR_TICKER,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            log_print("    No USD/INR data! Using fallback Rs.83.5")
            dates = pd.date_range(START_DATE, END_DATE, freq="D")
            return pd.DataFrame({"date": dates, "usdinr": 83.5})

        df.columns = flatten_columns(df.columns)
        df = df.reset_index()

        df.columns = [str(c).strip() for c in df.columns]
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})

        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date")

        close_col = "Close" if "Close" in df.columns else df.columns[1]
        df = df[["date", close_col]].rename(columns={close_col: "usdinr"})
        df["usdinr"] = pd.to_numeric(df["usdinr"], errors="coerce")
        df = df.dropna(subset=["usdinr"])

        df = (df.set_index("date")
                .resample("D")
                .ffill()
                .reset_index())

        df.to_csv(cache, index=False)
        log_print(f"    USD/INR saved: {len(df)} days | Latest: Rs.{df['usdinr'].iloc[-1]:.2f}")
        return df

    except Exception as e:
        log_print(f"    USD/INR failed: {e} | Using fallback Rs.83.5")
        dates = pd.date_range(START_DATE, END_DATE, freq="D")
        return pd.DataFrame({"date": dates, "usdinr": 83.5})


def flatten_columns(columns):
    """
    
    """
    if not isinstance(columns, pd.MultiIndex):
        return columns

    ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    lvl0 = set(columns.get_level_values(0))
    lvl1 = set(columns.get_level_values(1)) if columns.nlevels > 1 else set()

    if lvl0 & ohlcv:
        return columns.get_level_values(0)
    elif lvl1 & ohlcv:
        return columns.get_level_values(1)
    else:
        # Fallback: level 0, but at least we tried
        return columns.get_level_values(0)


# =============================================================================
# Convert USD -> INR
# =============================================================================
def convert_to_inr(price_usd, conversion_type, usdinr_rate):
    """
    USD price -> INR price as per Indian Standard
    Gold    : USD/oz     -> Rs/10g    (1oz = 31.1035g)
    Silver  : USD/oz     -> Rs/kg     (1kg = 32.1507 oz)
    Oil     : USD/bbl    -> Rs/bbl    (direct multiply)
    Gas     : USD/MMBtu  -> Rs/MMBtu  (direct multiply)
    Copper  : USD/lb     -> Rs/kg     (1kg = 2.20462 lb)
    """
    if conversion_type == "usd_oz_to_inr_10g":
        return price_usd * usdinr_rate / 3.11035

    elif conversion_type == "usd_oz_to_inr_kg":
        return price_usd * usdinr_rate * 32.1507

    elif conversion_type == "usd_to_inr":
        return price_usd * usdinr_rate

    elif conversion_type == "usd_lb_to_inr_kg":
        return price_usd * usdinr_rate * 2.20462

    return price_usd * usdinr_rate


# =============================================================================
# Fetch + Validate + Convert
# =============================================================================
def validate_data(df, name):
    issues = []
    if df is None or df.empty:
        return False, ["Empty dataframe"]
    if len(df) < 100:
        issues.append(f"Too few rows: {len(df)}")
    if "close_inr" in df.columns:
        if df["close_inr"].isnull().mean() > 0.1:
            issues.append("Too many nulls in close_inr")
        if (df["close_inr"] <= 0).any():
            issues.append("Non-positive prices found")
    if issues:
        log_print(f"    [{name}] Issues: {issues}")
    else:
        log_print(f"    [{name}] Validation passed!")
    return len(issues) == 0, issues


# =============================================================================
# Fetch Commodity
# =============================================================================
def fetch_commodity(name, info, df_usdinr):
    """Fetch + validate + convert to INR"""
    ticker = info["ticker"]
    log_print(f"\n  [{name}] Fetching {ticker}...")

    # Fail loud-but-clean if config is missing the conversion key,
    # instead of a bare KeyError deep inside convert_to_inr().
    if "conversion" not in info:
        log_print(
            f"  [{name}] FAILED: config/settings.py COMMODITIES['{name}'] "
            f"is missing a 'conversion' key. Check for a typo (e.g. "
            f"'convertion') or a missing entry."
        )
        return None

    try:
        raw = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            log_print(f"  [{name}] No data returned!")
            return None

        # KEY FIX: robust MultiIndex flatten (was the root cause of the
        # "cannot reindex on an axis with duplicate labels" failures)
        raw.columns = flatten_columns(raw.columns)

        raw = raw.reset_index()
        raw.columns = [str(c).strip() for c in raw.columns]

        if "Date" in raw.columns:
            raw = raw.rename(columns={"Date": "date"})
        elif "Datetime" in raw.columns:
            raw = raw.rename(columns={"Datetime": "date"})
        elif "index" in raw.columns:
            raw = raw.rename(columns={"index": "date"})

        raw["date"] = pd.to_datetime(raw["date"])

        raw = (raw.drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True))

        # Guard: if flatten_columns still left duplicate labels for any
        # reason, fail cleanly here rather than crashing later.
        if raw.columns.duplicated().any():
            dupes = raw.columns[raw.columns.duplicated()].tolist()
            log_print(f"  [{name}] FAILED: duplicate columns after flatten: {dupes}")
            return None

        col_map = {}
        for col in raw.columns:
            cl = col.lower()
            if cl == "open":
                col_map[col] = "open_usd"
            elif cl == "high":
                col_map[col] = "high_usd"
            elif cl == "low":
                col_map[col] = "low_usd"
            elif cl == "close":
                col_map[col] = "close_usd"
            elif cl == "volume":
                col_map[col] = "volume"
        raw = raw.rename(columns=col_map)

        df_usdinr_clean = df_usdinr.drop_duplicates(subset=["date"])
        raw = raw.merge(df_usdinr_clean[["date", "usdinr"]],
                        on="date", how="left")
        raw["usdinr"] = raw["usdinr"].ffill().fillna(83.5)

        conv = info["conversion"]
        for usd_col, inr_col in [
            ("open_usd", "open_inr"),
            ("high_usd", "high_inr"),
            ("low_usd", "low_inr"),
            ("close_usd", "close_inr"),
        ]:
            if usd_col in raw.columns:
                raw[inr_col] = convert_to_inr(
                    raw[usd_col], conv, raw["usdinr"]
                ).round(2)

        raw["daily_return"] = raw["close_inr"].pct_change()
        raw["price_range_inr"] = (raw.get("high_inr", raw["close_inr"]) -
                                raw.get("low_inr", raw["close_inr"]))

        raw["commodity"] = name
        raw["ticker"] = ticker
        raw["unit"] = info["unit"]
        raw["sector"] = info["sector"]

        is_valid, issues = validate_data(raw, name)

        latest = raw["close_inr"].iloc[-1]
        prev_val = raw["close_inr"].iloc[-2] if len(raw) > 1 else latest
        change = ((latest - prev_val) / prev_val * 100) if prev_val else 0
        direction = "UP" if change >= 0 else "DOWN"

        log_print(f"  [{name}] Done!")
        log_print(f"    Rows:   {len(raw):,}")
        log_print(f"    Period: {raw['date'].min().date()} -> {raw['date'].max().date()}")
        log_print(f"    Latest: {info['unit']} {latest:,.2f}  ({direction} {abs(change):.2f}%)")
        log_print(f"    Rate:   Rs.{raw['usdinr'].iloc[-1]:.2f}")

        return raw

    except Exception as e:
        log_print(f"  [{name}] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN
# =============================================================================
def fetch_all():
    """All commodities + exchange rate"""
    log_print("=" * 55)
    log_print("MULTI-COMMODITY DATA INGESTION (INR)")
    log_print(f"Period: {START_DATE} -> {END_DATE}")
    log_print("=" * 55)

    os.makedirs(RAW_PATH, exist_ok=True)

    df_usdinr = fetch_usdinr_rate()
    time.sleep(1)

    results = {}
    for name, info in COMMODITIES.items():
        df = fetch_commodity(name, info, df_usdinr)
        if df is not None:
            safe_name = name.replace(' ', '_').lower()
            path = f"{RAW_PATH}{safe_name}.csv"
            df.to_csv(path, index=False)
            log_print(f"  Saved: {path}")
            results[name] = df
        else:
            log_print(f"  [{name}] Skipped due to error")
        time.sleep(1)

    if results:
        df_all = pd.concat(list(results.values()), ignore_index=True)
        df_all.to_csv(f"{RAW_PATH}all_commodities.csv", index=False)
        log_print(f"\nAll combined: {RAW_PATH}all_commodities.csv")
        log_print(f"Shape: {df_all.shape}")

    return results, df_usdinr


def load_raw(name=None):
    if name:
        path = f"{RAW_PATH}{name.replace(' ', '_').lower()}.csv"
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"])
        return None
    path = f"{RAW_PATH}all_commodities.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None


def get_latest_prices():
    prices = {}
    for name, info in COMMODITIES.items():
        df = load_raw(name)
        if df is not None and len(df) > 1:
            df = df.sort_values("date")
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change = ((latest["close_inr"] - prev["close_inr"])
                      / prev["close_inr"] * 100)
            prices[name] = {
                "price": round(float(latest["close_inr"]), 2),
                "change_pct": round(float(change), 2),
                "date": str(latest["date"]),
                "unit": info["unit"],
                "icon": info.get("icon", ""),
                "direction": "up" if change >= 0 else "down",
            }
    return prices


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


    log_print("""
==========================================
MULTI-COMMODITY FORECASTER - INR Version
Step 1: Data Ingestion
==========================================
    """)

    results, df_usdinr = fetch_all()

    log_print("\n" + "=" * 55)
    log_print("SUMMARY - INR Prices:")
    log_print("=" * 55)
    for name, df in results.items():
        info = COMMODITIES[name]
        latest = df["close_inr"].iloc[-1]
        rate = df["usdinr"].iloc[-1]
        log_print(f"  [{name}] {info['unit']} {latest:>12,.2f}  (Rate: Rs.{rate:.2f})")

    log_print("""

    
Data Ingestion Complete! (INR)
Saved: data/raw/
Files: gold.csv, silver.csv, crude_oil.csv, natural_gas.csv,
    copper.csv, usdinr_rate.csv
    """)