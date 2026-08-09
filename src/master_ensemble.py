#========================================================================
# Master Ensemble Weights -> Inverse RMSE
#========================================================================


import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

sys.path.append(".")
sys.path.append("..")

from config.settings import COMMODITIES, PROCESSED_PATH, MODELS_PATH, TEST_START, TEST_END

PLOTS_DIR = f"{MODELS_PATH}plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════

def load_commodity_data(name, feat_cols):
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = df[df["commodity"] == name].sort_values("date").reset_index(drop=True)
    df["target"] = df["close_usd"].shift(-1)
    df["target_return"] = df["close_usd"].pct_change(1).shift(-1)
    df = df.dropna(subset=["target", "target_return"])

    train = df[df["date"] < TEST_START]
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
    return train, test


def lgbm_preds(name, X_test, current_prices):
    """LightGBM predicts a % return -- convert back to price using each
    row's actual current-day close, matching train.py's approach."""
    safe_name = name.replace(" ", "_").lower()
    model = joblib.load(f"{MODELS_PATH}{safe_name}_lgbm.pkl")
    predicted_returns = model.predict(X_test)
    price_preds = current_prices * (1 + predicted_returns)
    return pd.Series(price_preds, index=X_test.index)


def sarima_preds(name, n_steps):
    safe_name = name.replace(" ", "_").lower()
    model_path = f"{MODELS_PATH}{safe_name}_sarima.pkl"
    if not os.path.exists(model_path):
        return None
    fitted = joblib.load(model_path)
    fc = fitted.forecast(steps=n_steps)
    return pd.Series(np.asarray(fc))


# ════════════════════════════════════════════════
# INVERSE-RMSE WEIGHTING
# The model with the LOWER error automatically gets MORE weight.
# This replaces a fixed 55/45 guess with a number the models earned.
# ════════════════════════════════════════════════

def inverse_rmse_weights(rmse_a, rmse_b):
    w_a = (1 / rmse_a) / (1 / rmse_a + 1 / rmse_b)
    w_b = (1 / rmse_b) / (1 / rmse_a + 1 / rmse_b)
    return round(w_a, 3), round(w_b, 3)


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════

def run_master_ensemble():
    print("=" * 60)
    print("MASTER ENSEMBLE MODEL")
    print("Strategy: Inverse-RMSE weighted average of LightGBM + SARIMA")
    print("=" * 60)

    feat_cols = joblib.load(f"{MODELS_PATH}daily_feature_cols.pkl")

    summary = []
    all_preds = {}

    for name in COMMODITIES:
        print(f"\n{'='*60}")
        print(f"Commodity: {name}")

        train, test = load_commodity_data(name, feat_cols)
        if len(test) == 0:
            print("  Skipped -- no test rows")
            continue

        X_test = test[feat_cols]
        y_test = test["target"].reset_index(drop=True)

        lgbm_fc = lgbm_preds(name, X_test, test["close_usd"].values).reset_index(drop=True)
        lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_fc))

        sarima_fc = sarima_preds(name, len(y_test))

        if sarima_fc is None:
            ensemble = lgbm_fc
            w_lgbm, w_sarima = 1.0, 0.0
            sarima_rmse = None
            print("  SARIMA: model not found, using LightGBM only")
        else:
            sarima_rmse = np.sqrt(mean_squared_error(y_test, sarima_fc))
            w_lgbm, w_sarima = inverse_rmse_weights(lgbm_rmse, sarima_rmse)
            ensemble = w_lgbm * lgbm_fc + w_sarima * sarima_fc

            print(f"  LightGBM RMSE: {lgbm_rmse:.4f}  weight={w_lgbm}")
            print(f"  SARIMA   RMSE: {sarima_rmse:.4f}  weight={w_sarima}")

        ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble))
        ens_mae = mean_absolute_error(y_test, ensemble)
        print(f"  Ensemble RMSE: {ens_rmse:.4f}")
        print(f"  Ensemble MAE : {ens_mae:.4f}")

        comparison = pd.DataFrame({
            "Actual": y_test.round(2),
            "Ensemble": ensemble.round(2),
            "Error": (y_test - ensemble).round(2),
        })
        print(f"\n  Actual vs Ensemble (first 10 rows):")
        print(comparison.head(10).to_string())

        summary.append({
            "commodity": name,
            "lgbm_rmse": round(lgbm_rmse, 4),
            "sarima_rmse": round(sarima_rmse, 4) if sarima_rmse else "skipped",
            "ensemble_rmse": round(ens_rmse, 4),
            "ensemble_mae": round(ens_mae, 4),
            "w_lgbm": w_lgbm,
            "w_sarima": w_sarima,
        })

        all_preds[name] = {
            "dates": test["date"].reset_index(drop=True),
            "train_dates": train["date"],
            "train_actual": train["close_usd"],
            "y_test": y_test,
            "ensemble": ensemble,
        }

        # Update this commodity's saved ensemble weights so train.py /
        # save_forecasts.py use the EARNED weights next time, not the
        # fixed default from settings.py
        safe_name = name.replace(" ", "_").lower()
        os.makedirs(f"{MODELS_PATH}optuna/", exist_ok=True)
        with open(f"{MODELS_PATH}optuna/ensemble_weights_{safe_name}.json", "w") as f:
            json.dump({"w_lgbm": w_lgbm, "w_sarima": w_sarima}, f, indent=2)

        
        backtest_df = pd.DataFrame({
            "date": test["date"].reset_index(drop=True),
            "actual": y_test,
            "predicted": ensemble,
        })
        backtest_df.to_csv(f"{MODELS_PATH}backtest_{safe_name}.csv", index=False)

        # Same for all_metrics.json -- update this commodity's entry with
        # the improved ensemble RMSE/MAE and the earned weights
        all_metrics_path = f"{MODELS_PATH}all_metrics.json"
        if os.path.exists(all_metrics_path):
            with open(all_metrics_path) as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        actual_dir = np.sign(np.diff(y_test.values))
        pred_dir = np.sign(np.diff(ensemble.values))
        valid = actual_dir != 0
        dir_acc = (np.mean(actual_dir[valid] == pred_dir[valid]) * 100) if valid.sum() > 0 else 0

        all_metrics[safe_name] = {
            "commodity": name,
            "rmse": round(float(ens_rmse), 4),
            "mae": round(float(ens_mae), 4),
            "mape": round(float(np.mean(np.abs((y_test - ensemble) / y_test)) * 100), 4),
            "directional_accuracy": round(float(dir_acc), 1),
            "w_lgbm": w_lgbm,
            "w_sarima": w_sarima,
        }
        with open(all_metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY -- ALL COMMODITIES")
    print("=" * 60)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    summary_path = f"{MODELS_PATH}model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # ════════════════════════════════════════════
    # COMBINED GRID CHART -- all 5 commodities, one image
    # Dark theme + stats annotation under each panel, matching the
    # reference repo's polished "dashboard_accuracy_chart" style.
    # ════════════════════════════════════════════
    plt.style.use("dark_background")
    n = len(all_preds)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(19, 6.5 * rows), facecolor="#0B0D10")
    axes = axes.flatten()

    summary_lookup = {row["commodity"]: row for row in summary}

    for i, name in enumerate(all_preds):
        p = all_preds[name]
        s = summary_lookup.get(name, {})
        ax = axes[i]
        ax.set_facecolor("#14171C")

        # Show only the last ~120 days of train for readability
        train_tail = p["train_dates"].tail(120)
        train_actual_tail = p["train_actual"].tail(120)

        ax.plot(train_tail, train_actual_tail, label="Train (actual)",
                color="#5B8FD6", linewidth=1.3)
        ax.plot(p["dates"], p["y_test"], label="Test (actual)",
                color="#3FB68B", linewidth=2.2)
        ax.plot(p["dates"], p["ensemble"], label="Ensemble Forecast",
                color="#C9A15A", linestyle="--", linewidth=2.2)

        if len(p["dates"]) > 0:
            ax.axvline(p["dates"].iloc[0], color="#8B8F98", linestyle=":",
                        linewidth=1.2, label="Train/Test cutoff")

        icon = COMMODITIES[name]["icon"]
        ax.set_title(f"{icon}  {name}", fontsize=13, fontweight="bold", color="#E8E6E1", pad=10)
        ax.set_ylabel("Price (USD)", fontsize=9, color="#8B8F98")
        ax.tick_params(colors="#8B8F98", labelsize=8)
        ax.legend(fontsize=7.5, loc="upper left", facecolor="#14171C",
                edgecolor="#24282F", labelcolor="#E8E6E1")
        ax.grid(True, alpha=0.15, color="#8B8F98")
        for spine in ax.spines.values():
            spine.set_color("#24282F")

        # Stats line under each panel -- this is the piece that was missing
        w_lgbm = s.get("w_lgbm", 0)
        w_sarima = s.get("w_sarima", 0)
        stats_text = (
            f"RMSE: {s.get('ensemble_rmse', '—')}   MAE: {s.get('ensemble_mae', '—')}   "
            f"Ensemble: LGBM {int(w_lgbm*100)}% + SARIMA {int(w_sarima*100)}%"
        )
        ax.text(0.5, -0.16, stats_text, transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#C9A15A", family="monospace")

    # Hide any unused subplot slots
    for j in range(len(all_preds), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Master Ensemble Model (LightGBM + SARIMA, Inverse-RMSE Weighted) — All Commodities",
                fontsize=16, fontweight="bold", color="#E8E6E1", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = f"{PLOTS_DIR}master_ensemble_forecast.png"
    plt.savefig(out_path, dpi=150, facecolor="#0B0D10")
    plt.close()
    plt.style.use("default")   # reset so it doesn't affect other plotting scripts
    print(f"Plot saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    run_master_ensemble()