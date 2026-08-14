import sys, os
sys.path.append(".")
sys.path.append("..")

import pytest
import pandas as pd
import joblib
import json

from config.settings import COMMODITIES, PROCESSED_PATH, MODELS_PATH, RAW_PATH


# ════════════════════════════════════════════════
# 1. RAW DATA
# ════════════════════════════════════════════════

def test_raw_data():
    path = f"{RAW_PATH}all_commodities.csv"
    assert os.path.exists(path), f"Raw data missing: {path}. Run data_ingestion.py first."
    df = pd.read_csv(path)
    assert len(df) > 0, "Raw data file is empty"


def test_commodities():
    df = pd.read_csv(f"{RAW_PATH}all_commodities.csv")
    present = set(df["commodity"].unique())
    expected = set(COMMODITIES.keys())
    missing = expected - present
    assert not missing, f"Commodities missing from raw data: {missing}"


# ════════════════════════════════════════════════
# 2. PROCESSED FEATURES
# ════════════════════════════════════════════════

def test_processed_data():
    assert os.path.exists(PROCESSED_PATH), "Processed data missing. Run features.py first."


def test_feature_columns():
    path = f"{MODELS_PATH}daily_feature_cols.pkl"
    assert os.path.exists(path), "daily_feature_cols.pkl missing. Run features.py first."
    feat_cols = joblib.load(path)
    assert len(feat_cols) > 0


def test_cross_commodity_ratio():
    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")
    for ratio in ["gold_silver_ratio", "oil_gas_ratio", "copper_gold_ratio"]:
        assert ratio in feat_cols, f"{ratio} missing -- check commodity naming consistency"


def test_no_object_dtype_in_feature_columns():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")
    bad_cols = [c for c in feat_cols if df[c].dtype == "object"]
    assert not bad_cols, f"Non-numeric feature columns found: {bad_cols}"


# ════════════════════════════════════════════════
# 3. TRAINED MODELS
# ════════════════════════════════════════════════

@pytest.mark.parametrize("name", list(COMMODITIES.keys()))
def test_models(name):
    safe_name = name.replace(" ", "_").lower()
    assert os.path.exists(f"{MODELS_PATH}{safe_name}_lgbm.pkl"), f"Missing LGBM model for {name}"
    assert os.path.exists(f"{MODELS_PATH}{safe_name}_sarima.pkl"), f"Missing SARIMA model for {name}"


@pytest.mark.parametrize("name", list(COMMODITIES.keys()))
def test_models_are_loadable(name):
    safe_name = name.replace(" ", "_").lower()
    lgbm = joblib.load(f"{MODELS_PATH}{safe_name}_lgbm.pkl")
    sarima = joblib.load(f"{MODELS_PATH}{safe_name}_sarima.pkl")
    assert lgbm is not None and sarima is not None


# ════════════════════════════════════════════════
# 4. ENSEMBLE WEIGHTS
# ════════════════════════════════════════════════

@pytest.mark.parametrize("name", list(COMMODITIES.keys()))
def test_ensemble_weights_valid(name):
    from src.save_forecasts import load_ensemble_weights
    w = load_ensemble_weights(name)
    assert w["lgbm"] is not None, f"lgbm weight missing for {name}"
    assert w["sarima"] is not None, f"sarima weight missing for {name}"
    total = w["lgbm"] + w["sarima"]
    assert 0.95 <= total <= 1.05, f"Weights for {name} don't sum to ~1: {w}"


# ════════════════════════════════════════════════
# 5. FORECAST JSON (DB ki jagah ab isko check karte hain)
# ════════════════════════════════════════════════

FORECAST_JSON_PATH = "frontend/data/forecasts.json"


def test_forecast_json_exists():
    assert os.path.exists(FORECAST_JSON_PATH), \
        f"{FORECAST_JSON_PATH} missing. Run save_forecasts.py first."


def test_forecast_json_has_all_commodities():
    with open(FORECAST_JSON_PATH) as f:
        data = json.load(f)
    for name in COMMODITIES:
        safe_name = name.replace(" ", "_").lower()
        assert safe_name in data, f"{name} missing from forecasts.json"
        assert len(data[safe_name]["forecast"]) > 0, f"{name} has no forecast entries"


@pytest.mark.parametrize("name", list(COMMODITIES.keys()))
def test_forecast_prices_are_sane(name):
    """Regression: catches conversion bugs like Gold's '10g vs 10kg' mismatch
    that produced wildly inflated prices (the ₹3-lakh bug)."""
    safe_name = name.replace(" ", "_").lower()
    with open(FORECAST_JSON_PATH) as f:
        data = json.load(f)

    first_forecast = data[safe_name]["forecast"][0]
    price = first_forecast["ensemble_pred"]
    assert price > 0, f"{name} predicted price is not positive: {price}"

    sane_ranges = {
        "Gold": (30_000, 300_000),
        "Silver": (30_000, 400_000),
        "Crude Oil": (2_000, 15_000),
        "Natural Gas": (50, 800),
        "Copper": (300, 3_000),
    }
    low, high = sane_ranges.get(name, (0, float("inf")))
    assert low <= price <= high, f"{name} price {price} outside sane range ({low}-{high})"


@pytest.mark.parametrize("name", list(COMMODITIES.keys()))
def test_confidence_band_brackets_prediction(name):
    safe_name = name.replace(" ", "_").lower()
    with open(FORECAST_JSON_PATH) as f:
        data = json.load(f)

    first_forecast = data[safe_name]["forecast"][0]
    assert first_forecast["ci_low"] <= first_forecast["ensemble_pred"] <= first_forecast["ci_high"], \
        f"{name}: confidence band doesn't bracket predicted price -- {first_forecast}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])