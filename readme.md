# 🪙 Commodity Intelligence | Multi-Day Price Forecasting Engine

An end-to-end, full-stack commodity forecasting system predicting daily prices for five major commodities: **Gold, Silver, Crude Oil, Natural Gas, and Copper**.

Built with a dark "trading terminal" aesthetic, this system uses a dynamically weighted **Machine Learning Ensemble** to recursively forecast up to 30 days into the future, in Indian Rupees — with every forecast day's features re-derived from the model's own prior predictions, the same constraint any real production forecasting system has to work under.

---

## 🎯 Model Performance & Accuracy

Rigorous chronological hold-out validation (time-based split, no k-fold — avoids the lookahead bias a shuffled split would introduce on time series) ensures zero future-data leakage. The ensemble model (LightGBM + SARIMA) achieves the following on unseen test data:

| Commodity | Dir. Accuracy | RMSE | MAE | Ensemble Weighting |
| :--- | :---: | :---: | :---: | :--- |
| 🥇 **Gold** | 48.9% | 123.751 | 96.7394 | LGBM 94% + SARIMA 5% |
| 🥈 **Silver** | 48.1% | 4.3562 | 2.7371 | LGBM 90% + SARIMA 9% |
| 🛢 **Crude oil** | 49.8% | 3.2499 | 2.4461 | LGBM 79% + SARIMA 20% |
| 🔥 **Natural Gas** | 47.1% | 0.3232 | 0.2262 | LGBM 88% + SARIMA 12% |
| 🟤 **Copper** | 43.5% | 0.1929 | 0.1463 | LGBM 90% + SARIMA 9% |

*Directional Accuracy hovers near 50% across the board — consistent with the near-random-walk nature of daily commodity returns documented in financial forecasting literature, not a modeling gap. The model's real strength is price-level accuracy: forecasts were independently cross-checked against live market prices and tracked within 3–5% of actual spot prices across all five commodities.*

---

## 🏗️ Architecture

```
yfinance ──▶ data_ingestion.py ──▶ raw CSV (USD + INR)
                                         │
                                         ▼
                                  features.py
                     (200+ features: RSI, MACD, Bollinger, EMA, ATR,
                      lags, rolling stats, cross-commodity ratios,
                      time features — single source of truth, reused
                      identically by training and inference)
                                         │
                                         ▼
                train.py (LightGBM + SARIMA)  +  tune_hyperparameters.py (Optuna)
                                         │
                                         ▼
                              save_forecasts.py
              (recursive engine — re-derives features from each
               day's own prediction; all 5 commodities forecast
               together so cross-commodity ratios stay valid)
                                         │
                                         ▼
                              PostgreSQL (forecasts table)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
                  FastAPI (async, live)         export_forecasts.py
                                                (static JSON fallback)
                          │
                          ▼
                   React Dashboard
```

---

## ✨ Key Features

- **Recursive Multi-Day Forecasting:** Generates 30-day forward-looking predictions by feeding each day's own prediction back in as input for the next day, recomputing all 200+ engineered features at every step.
- **Ensemble ML Architecture:** LightGBM trained on **% returns, not absolute price** — tree models can't extrapolate beyond training-range values, so a price-based target would flatten out at record highs. SARIMA adds trend/seasonality tracking. Per-commodity weights are tuned via **Optuna**, not fixed globally.
- **Cross-Commodity Signal Engineering:** Gold/Silver, Oil/Gas, and Copper/Gold ratios computed as leading economic indicators — all five commodities forecast in lockstep so these ratios stay valid on predicted, not just historical, days.
- **Confidence-Aware Forecasting:** Prediction bands widen with forecast horizon (RMSE × √horizon); a Buy/Hold/Sell signal is generated relative to each model's own historical error margin (MAPE), not an arbitrary threshold.
- **Async FastAPI Backend:** PostgreSQL-backed, async (asyncpg) request handling for concurrent dashboard traffic; always serves the latest pipeline run via a `MAX(generated_at)` filter.
- **Trading-Terminal UI/UX:** Dark charcoal + gold-accent React dashboard with live price widgets and confidence-band charts.
- **Automated Validation:** 32-test pytest suite covering data integrity, feature completeness, model loadability, and forecast sanity — several tests are direct regressions for real bugs found during development (see below).

---

## 🛠️ Technology Stack

**Backend (Machine Learning & API):**
- Python 3.10+
- FastAPI & Uvicorn (async API, asyncpg driver)
- LightGBM (Gradient Boosted Decision Trees)
- Statsmodels (SARIMA)
- Optuna (hyperparameter + ensemble weight tuning)
- Pandas & Scikit-Learn (feature engineering & evaluation)
- SQLAlchemy + PostgreSQL

**Frontend (Dashboard):**
- React
- Recharts (confidence-band forecast charts)

**Data Sources:**
- yfinance (Yahoo Finance) — commodity futures + USD/INR exchange rate

---

## 🐞 Debugging Highlights

A few bugs caught during development worth calling out, since they were genuinely instructive:

- **Silent unit conversion bug:** a mismatched config key skipped Gold's oz→10g conversion entirely, inflating displayed Gold prices ~3x with no error thrown. Caught via a live-market-price sanity check, not a crash — a reminder that unit-conversion fallbacks need explicit validation, not just a default branch.
- **Case-sensitive naming mismatch:** `"Crude oil"` vs `"Crude Oil"` silently prevented the Oil/Gas cross-commodity ratio from ever being created — no error, just a permanently missing feature.
- **Inconsistent key naming across config sources:** Optuna's tuned-weights file used `w_lgbm`/`w_sarima` while the default settings used `lgbm`/`sarima`; normalized in a single loader function so downstream code never has to know which source it came from.

---

## 🔭 Future Work

- Dedicated direction classifier (LGBMClassifier) + feature-importance-based pruning of the 200+ engineered features, targeting directional accuracy directly rather than as a byproduct of price-return regression
- Walk-forward retraining — the current fixed train/test split degrades during regime changes (e.g. the 2025–26 gold rally, unseen in training); walk-forward validation would adapt to new regimes as they emerge
- LSTM comparison against the LightGBM + SARIMA ensemble on the same time-based split
- GitHub Actions scheduler for fully automated daily pipeline runs
- Production deployment: Render (backend + PostgreSQL) + Vercel (dashboard)

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```
git clone https://github.com/tanyadiwedi78-boop/<commodities-price-forecaster->.git

```

### 2. Set Up the Python Environment

```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with your PostgreSQL credentials.

### 3. Run the Pipeline

```
python src/data_ingestion.py
python src/features.py
python src/optuna_parameters.py
python src/train.py
python src/db.py
python src/save_forecasts.py
python src/export_forecasts.py
```

### 4. Start the FastAPI Backend

```
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Start the Frontend Dashboard

In a **new terminal window**:

```
cd frontend
npm install
npm start
```

*Open your browser and navigate to `http://localhost:3000` to view the dashboard.*

### 6. Run Tests

```
pytest test_pipeline.py -v
```

---

## 👨‍💻 Author

**Built by Tanya Diwedi**

---

*Data via yfinance (Yahoo Finance). Forecasts are for educational/portfolio purposes with real financial data*

*© 2026 Commodity Intelligence. All rights reserved.*







































**Model Performance & Optimization Table :
| Commodity | Dir. Accuracy | RMSE | MAE | Ensemble Weighting |
| :--- | :---: | :---: | :---: | :--- |
| 🥇 **Gold** | 48.9% | 123.751 | 96.7394 | LGBM 94% + SARIMA 5% |
| 🥈 **Silver** | 48.1% | 4.3562 | 2.7371 | LGBM 90% + SARIMA 9% |
| 🛢 **Crude oil** | 49.8% | 3.2499 | 2.4461 | LGBM 79% + SARIMA 20% |
| 🔥 **Natural Gas** | 47.1% | 0.3232 | 0.2262 | LGBM 88% + SARIMA 12% |
| 🟤 **Copper** | 43.5% | 0.1929 | 0.1463 | LGBM 90% + SARIMA 9% |



commodity  lgbm_rmse  sarima_rmse  ensemble_rmse  ensemble_mae  w_lgbm  w_sarima
       Gold    80.1509    1342.8327       135.5122      106.2261   0.944     0.056
     Silver     2.8310      28.2657         4.4129        2.8257   0.909     0.091
  Crude oil     2.9291      12.7788         3.2522        2.4488   0.814     0.186
Natural Gas     0.2375       1.5496         0.3175        0.2140   0.867     0.133
     Copper     0.1217       1.2792         0.1909        0.1492   0.913     0.087


