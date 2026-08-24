#  Commodity Intelligence — Live Multi-Commodity Price Forecaster

A full-stack, automated MLOps system that forecasts prices for five commodities — **Gold, Silver, Crude Oil, Natural Gas, and Copper** — up to 30 days ahead, using a LightGBM + SARIMA ensemble served live through an in-memory FastAPI backend.


---

## 📸 Dashboard

![Commodity Intelligence Dashboard](docs/dashboard-screenshot.png)

Each card shows a live price, day-over-day change, and a mini sparkline against the previous close. Clicking a card expands into a full chart with history, a 30-day recursive forecast, a confidence band, and a timeframe selector (1D → Max).

---

## 🏗️ Architecture

```
GitHub Actions (scheduled)
   ├── daily (Mon–Fri): data_ingestion.py → features.py
   │       fresh raw + processed data committed back to the repo
   └── weekly (Sunday): + train.py
           retrained LightGBM/SARIMA models committed back to the repo
                              │
                              ▼
                    api/main.py (FastAPI)
        On startup (lifespan event, once):
          • loads raw data + trained models from the repo
          • runs the recursive 30-day forecast for all 5 commodities
          • caches the result in memory
        Every request after that:
          • served instantly from the in-memory cache — no recompute,
            no disk I/O, no database round-trip
                              │
                              ▼
              GET /forecasts  (single endpoint, full dump)
                              │
                              ▼
                frontend/  (vanilla HTML + CSS + JS)
        fetch()es the live API directly, renders 5 cards with
        Chart.js sparklines, and an expandable full-forecast modal
```

**Why this design:** the recursive forecast (30 days × 5 commodities, with cross-commodity features recomputed at every step) takes a few seconds to compute — too slow to redo on every request, unnecessary to hit a database for. Computing it once at server startup and serving from memory gives near-instant response times, at the cost of the forecast only refreshing when the server restarts (which happens automatically on every GitHub Actions push, via Render's auto-deploy).

---

## ✨ Key Features

- **Recursive Multi-Day Forecasting** — generates a 30-day forward forecast by feeding each day's own prediction back in as input for the next day, recomputing all 200+ engineered features at every step (RSI, MACD, Bollinger Bands, EMA, ATR, lag/rolling features, cross-commodity ratios) — the same constraint any real production forecaster has to work under, since future data doesn't exist yet.
- **Ensemble Model** — LightGBM trained on **% returns, not absolute price** (tree models can't extrapolate past training-range values, so a price-based target flattens out at record highs — which Gold and Silver both hit during this project's test period). SARIMA adds trend/seasonality tracking. Per-commodity ensemble weights are tuned via **Optuna**, not fixed globally.
- **Cross-Commodity Signal Engineering** — Gold/Silver, Oil/Gas, and Copper/Gold ratios, computed as leading economic indicators. All five commodities are forecast in lockstep so these ratios stay valid on predicted (not just historical) days.
- **Confidence-Aware Forecasting** — prediction bands widen with forecast horizon (RMSE × √horizon), visualized directly on the expanded chart.
- **In-Memory Model Serving** — FastAPI loads models once at startup and serves every request from RAM. No database, no per-request recomputation.
- **Fully Automated Pipeline** — GitHub Actions runs data ingestion daily and full retraining weekly, committing fresh data/models back to the repo, which triggers an automatic redeploy.
- **32-Test Validation Suite** — covers data integrity, feature completeness, model loadability, ensemble weight consistency, and forecast sanity bounds. Several tests are direct regressions for real bugs found during development.

---

## 🛠️ Tech Stack

**Backend / ML:**
- Python 3.10
- FastAPI + Uvicorn (in-memory model-serving API)
- LightGBM (gradient boosted trees)
- Statsmodels (SARIMA)
- Optuna (hyperparameter + ensemble weight tuning)
- Pandas, NumPy, scikit-learn, joblib

**Frontend:**
- Vanilla HTML / CSS / JavaScript (no framework, no build step)
- Chart.js (sparklines + full forecast charts)

**Automation & Hosting:**
- GitHub Actions (scheduled daily/weekly pipeline runs)
- Render (backend hosting, native Python runtime — no Docker)
- Vercel / static hosting (frontend)

**Data Source:**
- yfinance (Yahoo Finance) — commodity futures + USD/INR exchange rate

---

## 🎯 Model Performance

Evaluated on a chronological hold-out split:

| Commodity | RMSE | MAPE | Directional Accuracy | Ensemble Weighting |
|---|---:|---:|---:|---|
| 🥇 Gold | 123.75 | 2.44% | 48.9% | LightGBM 94.7% + SARIMA 5.3% |
| 🥈 Silver | 4.36 | 4.43% | 48.1% | LightGBM 90.7% + SARIMA 9.3% |
| 🛢 Crude Oil | 3.25 | 3.42% | 49.8% | LightGBM 79.6% + SARIMA 20.4% |
| 🔥 Natural Gas | 0.32 | 6.36% | 47.1% | LightGBM 88.0% + SARIMA 12.0% |
| 🟤 Copper | 0.19 | 2.71% | 43.5% | LightGBM 90.3% + SARIMA 9.7% |

**On directional accuracy:** ~45–50% is consistent with the near-random-walk nature of daily commodity returns, a well-documented limit in financial forecasting — not a modeling gap. The model's real strength is price-level accuracy (MAPE 2–6%), which is what the recursive forecast and confidence bands are built around.

**On live validation:** forecasts were cross-checked against real market prices during development and tracked within ~3–6% of actual spot prices. The largest gaps appeared during a sharp, news-driven Gold/Silver rally in August 2026 — a regime the training data hadn't seen, causing the model to under-predict the magnitude of the move while still calling the correct direction. This is a known limitation of any fixed train/test split; see [Future Work](#-future-work).

---

## 🔭 Future Work

- **Walk-forward retraining** — the current fixed train/test split degrades during regime changes (e.g. the August 2026 gold rally, unseen in training); walk-forward validation would adapt to new regimes as they emerge.
- **Directional accuracy** — a dedicated LGBMClassifier for up/down prediction, plus feature-importance-based pruning of the 200+ engineered features, targeting directional accuracy directly rather than as a byproduct of return regression.
- **LSTM comparison** — evaluate against the LightGBM + SARIMA ensemble on the same time-based split.

---

## 🚀 Setup

### 1. Clone and install

```bash
git clone https://github.com/tanyadiwedi78-boop/commodity-price-forecaster-.git
cd commodity-price-forecaster-

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Run the pipeline (generates raw data, features, and trained models)

```bash
python src/data_ingestion.py
python src/features.py
python src/train.py
```

### 3. Start the backend

```bash
uvicorn api.main:app --reload --port 8000
```

Wait for `Cached forecasts for 5 commodities. Server ready.` — the first startup computes the recursive forecast and takes a few seconds.

### 4. Open the dashboard

Open `frontend/index.html` with a local server (e.g. VS Code's Live Server extension), or:

```bash
cd frontend
python -m http.server 8080
```

Visit `http://localhost:8080`.

### 5. Run tests

```bash
pytest test_pipeline.py -v
```

---

## 📁 Project Structure

```
├── config/settings.py          # commodities, model params, feature config
├── src/
│   ├── data_ingestion.py       # yfinance fetch + USD→INR conversion
│   ├── features.py             # feature engineering (single source of truth)
│   ├── train.py                # LightGBM + SARIMA training
│   ├── tune_hyperparameters.py # Optuna hyperparameter search
│   ├── save_forecasts.py       # recursive multi-day forecast engine
│   │                             (functions reused directly by api/main.py)
│   └── db.py                   # PostgreSQL schema (built + tested, not
│                                  used in the deployed architecture)
├── api/
│   └── main.py                 # FastAPI, in-memory model serving
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/dashboard.js
├── models/                     # trained models, metrics, backtest charts
├── test_pipeline.py            # 32-test validation suite
├── .github/workflows/          # daily forecast + weekly retrain automation
└── requirements.txt
```

---

## 👨‍💻 Author

**Tanya Diwedi**

---

*Data via [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance). Forecasts are for educational/portfolio purposes only — not financial advice.*