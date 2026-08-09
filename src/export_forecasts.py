import pandas as pd
import numpy as np
import json, os, sys
from sqlalchemy import text

sys.path.append(".")
sys.path.append("..")

from config.settings import COMMODITIES, PROCESSED_PATH, MODELS_PATH
from src.db import get_engine
from src.data_ingestion import convert_to_inr

OUTPUT_PATH = "frontend/data/forecasts.json"
HISTORY_DAYS = 1825   # ~5 years -- enough for the dashboard's timeframe selector


def load_history(name):
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = df[df["commodity"] == name].sort_values("date").tail(HISTORY_DAYS)

    conv_type = COMMODITIES[name].get("conversion") or COMMODITIES[name].get("Conversion")
    has_rate = "usdinr" in df.columns

    result = {}
    for _, row in df.iterrows():
        price_usd = float(row["close_usd"])
        if conv_type and has_rate and pd.notna(row["usdinr"]):
            price = convert_to_inr(price_usd, conv_type, float(row["usdinr"]))
        else:
            price = price_usd  # fallback -- no conversion info available for this row
        result[row["date"].strftime("%Y-%m-%d")] = round(price, 2)
    return result


def load_forecast(name, engine):
    query = text("""
        SELECT forecast_date, predicted_price, predicted_price_lgbm,
            predicted_price_sarima, ci_low, ci_high, horizon_days,
            recommendation, reasoning, pct_change
        FROM forecasts
        WHERE commodity = :name
        ORDER BY generated_at DESC, horizon_days ASC
        LIMIT 30
    """)
    df = pd.read_sql(query, engine, params={"name": name})
    if df.empty:
        return []

    return [
        {
            "date": str(row["forecast_date"]),
            "horizon_days": int(row["horizon_days"]),
            "ensemble_pred": float(row["predicted_price"]),
            "lgbm_pred": float(row["predicted_price_lgbm"]) if pd.notna(row["predicted_price_lgbm"]) else None,
            "sarima_pred": float(row["predicted_price_sarima"]) if pd.notna(row["predicted_price_sarima"]) else None,
            "ci_low": float(row["ci_low"]) if pd.notna(row["ci_low"]) else None,
            "ci_high": float(row["ci_high"]) if pd.notna(row["ci_high"]) else None,
            "recommendation": row["recommendation"] if pd.notna(row["recommendation"]) else None,
            "reasoning": row["reasoning"] if pd.notna(row["reasoning"]) else None,
            "pct_change": float(row["pct_change"]) if pd.notna(row["pct_change"]) else None,
        }
        for _, row in df.iterrows()
    ]

def load_metrics(name):
    safe_name = name.replace(" ", "_").lower()
    path = f"{MODELS_PATH}all_metrics.json"
    if not os.path.exists(path):
        return {"rmse": None, "mape": None, "w_lgbm": None, "w_sarima": None}
    with open(path) as f:
        all_metrics = json.load(f)
    m = all_metrics.get(safe_name, {})
    return {
        "rmse": m.get("rmse"),
        "mape": m.get("mape"),
        "w_lgbm": m.get("w_lgbm"),
        "w_sarima": m.get("w_sarima"),
    }


def export_all():
    print("=" * 55)
    print("EXPORTING forecasts.json FOR DASHBOARD")
    print("=" * 55)

    engine = get_engine()
    output = {}

    for name in COMMODITIES:
        safe_name = name.replace(" ", "_").lower()
        print(f"  Exporting: {name}")

        output[safe_name] = {
            "name": name,
            "unit": COMMODITIES[name]["unit"],
            "icon": COMMODITIES[name]["icon"],
            "history": load_history(name),
            "forecast": load_forecast(name, engine),
            "metrics": load_metrics(name),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_all()