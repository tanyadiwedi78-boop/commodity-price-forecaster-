import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys, json

sys.path.append(".")
sys.path.append("..")

from config.settings import COMMODITIES, MODELS_PATH

PLOTS_DIR = f"{MODELS_PATH}plots/"


def plot_commodity_backtest(name):
    safe_name = name.replace(" ", "_").lower()
    backtest_path = f"{MODELS_PATH}backtest_{safe_name}.csv"
    metrics_path = f"{MODELS_PATH}all_metrics.json"

    if not os.path.exists(backtest_path):
        print(f"  Skipped {name} -- {backtest_path} not found. Run train.py first.")
        return

    df = pd.read_csv(backtest_path, parse_dates=["date"])

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
        metrics = all_metrics.get(safe_name, {})

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0B0D10")
    ax.set_facecolor("#14171C")

    ax.plot(df["date"], df["actual"], label="Actual Price", color="#3FB68B", linewidth=2.2)
    ax.plot(df["date"], df["predicted"], label="Predicted (Ensemble)", color="#C9A15A",
            linewidth=2.2, linestyle="--")
    ax.fill_between(df["date"], df["actual"], df["predicted"],
                    color="#C9A15A", alpha=0.10)

    icon = COMMODITIES[name]["icon"]
    title = f"{icon}  {name} — Actual vs Predicted (Test Period)"

    # FIGURE-level title (sabse upar, bold, bada) -- ab ye ax ke bahar hai,
    # isliye axes ke apne title/text ke saath kabhi overlap nahi karega
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#E8E6E1", y=0.98)

    if metrics:
        subtitle = (f"RMSE: {metrics.get('rmse', '—')}   MAE: {metrics.get('mae', '—')}   "
                    f"MAPE: {metrics.get('mape', '—')}%   Dir. Accuracy: {metrics.get('directional_accuracy', '—')}%")
        # AXES-level subtitle -- ab ye normal ax.set_title hai, apni jagah pe hi
        # rehta hai (axes ke top), suptitle se kaafi neeche kyunki suptitle
        # figure-level hai, axes se bahar render hota hai
        ax.set_title(subtitle, fontsize=10, color="#8B8F98", family="monospace", pad=10)

    ax.set_ylabel("Price (USD)", fontsize=10, color="#8B8F98")
    ax.tick_params(colors="#8B8F98", labelsize=9)
    ax.legend(fontsize=10, loc="upper left", facecolor="#14171C",
            edgecolor="#24282F", labelcolor="#E8E6E1")
    ax.grid(True, alpha=0.15, color="#8B8F98")
    for spine in ax.spines.values():
        spine.set_color("#24282F")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    # rect ka top=0.90 rakha -- taaki suptitle ke liye upar jagah bachi rahe,
    # tight_layout usme axes/subtitle ko squeeze na kare
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = f"{PLOTS_DIR}backtest_{safe_name}.png"
    plt.savefig(out_path, dpi=150, facecolor="#0B0D10")
    plt.close()
    plt.style.use("default")
    print(f"  Saved: {out_path}")

def plot_all_backtests():
    print("=" * 55)
    print("GENERATING ACTUAL vs PREDICTED BACKTEST CHARTS")
    print("=" * 55)

    for name in COMMODITIES:
        plot_commodity_backtest(name)

    print(f"\nAll backtest charts saved to: {PLOTS_DIR}")


def build_results_table():
    """Builds a Markdown table (paste straight into README.md) summarizing
    every commodity's test-set performance, matching the reference repo's
    'Model Performance & Optimization' table."""
    metrics_path = f"{MODELS_PATH}all_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"  {metrics_path} not found. Run train.py first.")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    rows = []
    for name in COMMODITIES:
        safe_name = name.replace(" ", "_").lower()
        m = all_metrics.get(safe_name)
        if not m:
            continue
        icon = COMMODITIES[name]["icon"]
        rows.append(
            f"| {icon} **{name}** | {m['directional_accuracy']}% | {m['rmse']} | "
            f"{m['mae']} | LGBM {int(m['w_lgbm']*100)}% + SARIMA {int(m['w_sarima']*100)}% |"
        )

    table = (
        "| Commodity | Dir. Accuracy | RMSE | MAE | Ensemble Weighting |\n"
        "| :--- | :---: | :---: | :---: | :--- |\n"
        + "\n".join(rows)
    )
    print("\n" + "=" * 55)
    print("README RESULTS TABLE (copy-paste this into README.md)")
    print("=" * 55)
    print(table)

    with open(f"{MODELS_PATH}results_table.md", "w") as f:
        f.write(table)
    print(f"\nAlso saved to: {MODELS_PATH}results_table.md")


if __name__ == "__main__":
    plot_all_backtests()
    build_results_table()