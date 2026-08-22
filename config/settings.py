from datetime import datetime , timedelta
import os
from dotenv import load_dotenv
load_dotenv()

#----------------------Currency--------------------------------------#
CURRENCY        = "INR"
CURRENCY_SYMBOL =  "₹"
USDINR_TICKER   = "INR=X"

#----------------------Commodities-----------------------------------#
COMMODITIES = {
    "Gold" : {
        "ticker"      : "GC=F",
        "name"        : "Gold",
        "unit"        : "₹/10g",
        "full_unit"   : "Rupees per 10 gram",
        "icon"        : "🥇",
        "sector"      : "Precious metal",
        "why"         : "Safe haven , inflation hedge",
        "Key_drivers" : ["USD/INR Rate" , "Fed Rates"],
        # Conversion: USD/oz --> INR/10g
        # 1 oz = 31.1035 grams
        # INR/10g = USD/oz * USDINR_Rate / 3.11035
        "conversion"  : "usd_oz_to_inr_10g",
    },
    
    "Silver" : {
        "ticker" : "SI=F",
        "name"   : "Silver",
        "unit"   : "₹/kg" , 
        "unit_full" : "Rupees per kilograms",
        "icon"      : "🥈" , 
        "sector"    : "Precious Metal",
        "why"       : "Industrial + precious metal" ,
        "Key_drivers" : ["Gold Price" ,"Industrial Demand"],
        "conversion"  : "usd_oz_to_inr_kg", 
    },

    "Crude oil" : {
        "ticker" : "CL=F",
        "name"   : "Crude oil",
        "unit"   : "₹/barrel",
        "unit_full" : "Rupees per barrel",
        "icon"      : "🛢",
        "sector"    : "Energy",
        "why"       : "Winter demand seasonal" ,
        "Key_drivers" : ["OPEC Supply" , "USD/INR" ,"Global GDP"], 
        "conversion" : "usd_to_inr" ,
    },

    "Natural Gas" : {
        "ticker" : "NG=F",
        "name"   : "Natural Gas",
        "unit"   : "₹/kg",
        "unit_full" : "₹/MMBtu",
        "icon"      : "🔥",
        "sector"    : "Energy" ,
        "why"       : "Winter demand seasonal",
        "Key_drivers" : ["China Demand" , "USD/INR" , "Construction"],
        "conversion"  : "usd_to_inr",
    },

    "Copper" : {
        "ticker" : "HG=F",
        "name"   : "Copper",
        "unit"   : "₹/kg",
        "unit_full" : "Rupees per kilogram",
        "icon"      : "🟤" ,
        "sector"    : "Industrial metal" ,
        "why"       : "Economic leading indicator",
        "Key_drivers" : ["weather" , "Storage" ,"USD/INR"],
        "conversion" : "usd_lb_to_inr_kg",
    },

}

#------------------------- Date Settings ----------------------------#
START_DATE     = "2010-01-01"
END_DATE       = datetime.today().strftime("%Y-%m-%d")
TEST_START     = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")  # last 2 years
TEST_END       = datetime.today().strftime("%Y-%m-%d")
FORCAST_MONTHS =36 # 3 Years Monthly
FORCAST_DAYS   =30 # 30 days daily

#-------------------------- Paths-----------------------------------#
RAW_PATH         = "data/raw/"
PROCESSED_PATH   = "data/processed/merged_features.csv"
MODELS_PATH      = "models/"
FORCAST_JSON     = "dashboard/data/forecasts.json"
LOG_PATH         = "logs/"

#---------------------------ML SETTINGS -----------------------------#
LGBM_PARAMS = {
    "objectives"   : "regression",
    "metric"       : "rmse",
    "n_estimators" : 1000,
    "learning_rate": 0.03,
    "num_leaves"   : 63,
    "verbose"      : -1,
    "random_state" : 42,
}
SARIMA_ORDER    = (1,1,1)
SARIMA_SEASONAL  = (1,1,1,12)
ENSEMBLE_WEIGHTS = {"lgbm" : 0.55 , "sarima" : 0.45}

#------------------------- TECHNICAL INDICATOR---------------------------#
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD    = 2
EMA_PERIOD       = [9 , 21 , 50 , 200]
ATR_PERIOD       = 14
ROLLING_WINDOWS  = [7 , 14 , 30 , 60 , 90 , 180]


#------------------ DASHBOARD ------------------------------------#
CHART_COLORS = { 
    "history"  : "#3b82f6" , 
    "forecast" : "#22c55e" ,
    "ci_band"  : "rgba(34,197,94,0.15)" , 
    "bullish"  : "#22c55e" ,
    "bearish"  : "#ef4444" ,
} 





def get_confidence_label(accuracy):
    if accuracy >= 80 :   return "High Confidence"
    elif accuracy >= 70 : return "Medium Confidence"
    else :                return "low Confidence"




API_HOST  = "0.0.0.0"
API_PORT  = "8000"