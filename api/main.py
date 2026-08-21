import sys , os
sys.path.append(".")
sys.path.append("..")

from contextlib import asynccontextmanager
from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware

import joblib
from config.settings import COMMODITIES , MODELS_PATH
from src.data_ingestion import load_raw
from src.save_forecasts import (
    generate_all_forecasts_recursive ,
    load_history ,
    load_metrics ,
)

CACHE = {}

@asynccontextmanager
async def lifespan(app : FastAPI):
    print("Loading data + models into memory , computing forecasts...")

    df_all = load_raw()
    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")
    forecasts_by_commodity = generate_all_forecasts_recursive(df_all, feat_cols)

    output = {}
    for name in COMMODITIES:
        safe_name = name.replace(" " , "_").lower()
        output[safe_name] = {
            "name" : name,
            "unit" : COMMODITIES[name]["unit"],
            "icon" : COMMODITIES[name]["icon"],
            "history" : load_history(name),
            "forecast" : forecasts_by_commodity[name],
            "metrics" : load_metrics(name) ,
        }
    
    CACHE["forecasts"] = output
    print(f"Cached forecasts for {len(output)} commodities. Server ready")

    yield

    CACHE.clear()
    print("Server shutting down , cache cleared")


app = FastAPI(title = "Multi-Commodity Forecaster API" , lifespan = lifespan)

app.add_middleware(
    CORSMiddleware ,
    allow_origins = [
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",      # ← Live Server default address,
        "https://YOUR-APP-NAME.vercel.app",
    ],
        
    allow_methods = ["GET"],
    allow_headers = ["*"],
)
@app.get("/")
def root():
    return {"status": "ok", "commodities": list(CACHE.get("forecasts", {}).keys())}


@app.get("/forecasts")
def get_all_forecasts():
    if not CACHE.get("forecasts"):
        raise HTTPException(status_code=503, detail="Forecasts not ready yet, server still starting up")
    return CACHE["forecasts"]


@app.get("/forecasts/{commodity}")
def get_one_forecast(commodity: str):
    data = CACHE.get("forecasts", {})
    if commodity not in data:
        raise HTTPException(status_code=404, detail=f"Unknown commodity '{commodity}'. Valid: {list(data.keys())}")
    return data[commodity]
