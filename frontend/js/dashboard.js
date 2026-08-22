// ============================================================
// COMMODITY INTELLIGENCE — DASHBOARD ENGINE
// ============================================================

const DATA_SOURCES = ["data/forecasts.json", "http://localhost:8000/forecasts"];

let allData = {};        // raw JSON keyed by safe_name
let commodities = [];    // processed array
let sparkCharts = {};    // per-card sparkline charts
let detailChart = null;  // main detail chart instance
let activeCommodity = null;
let activeTimeframe = "1M";

// ============================================================
// DATA LOADING (static JSON first, API fallback)
// ============================================================
async function loadForecasts() {
  for (const url of DATA_SOURCES) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const raw = await res.json();
      return raw;
    } catch (_) { /* try next */ }
  }
  throw new Error("Could not load forecast data from any source");
}

function processData(raw) {
  allData = raw;
  commodities = Object.values(raw).map((c) => {
    const safeName = Object.keys(raw).find(k => raw[k] === c);
    const historyEntries = Object.entries(c.history).sort(([a], [b]) => a.localeCompare(b));
    const last = historyEntries[historyEntries.length - 1];
    const prev = historyEntries[historyEntries.length - 2];

    const price = last ? last[1] : 0;
    const prevClose = prev ? prev[1] : price;
    const changePct = prevClose ? ((price - prevClose) / prevClose) * 100 : 0;

    const sparkline = historyEntries.slice(-30).map(([d, p]) => ({ date: d, price: p }));
    const fullHistory = historyEntries.map(([d, p]) => ({ date: d, price: p }));

    const forecast = (c.forecast || []).map((f) => ({
      date: f.date,
      horizon: f.horizon_days,
      ensemble: f.ensemble_pred,
      lgbm: f.lgbm_pred,
      sarima: f.sarima_pred,
      ciLow: f.ci_low,
      ciHigh: f.ci_high,
      recommendation: f.recommendation,
      reasoning: f.reasoning,
      pctChange: f.pct_change,
    }));

    const metrics = c.metrics || {};
    const firstForecast = forecast[0];

    return {
      safeName, name: c.name, icon: c.icon, unit: c.unit,
      price, prevClose, changePct,
      sparkline, fullHistory, forecast,
      metrics,
      recommendation: firstForecast ? firstForecast.recommendation : "HOLD",
      reasoning: firstForecast ? firstForecast.reasoning : "",
    };
  });
}

// ============================================================
// SUMMARY BAR
// ============================================================
function renderSummary() {
  const mapes = commodities.map(c => c.metrics.mape).filter(Boolean);
  const rmses = commodities.map(c => c.metrics.rmse).filter(Boolean);
  const avgMape = mapes.length ? (mapes.reduce((a, b) => a + b, 0) / mapes.length).toFixed(2) : "—";
  const avgRmse = rmses.length ? (rmses.reduce((a, b) => a + b, 0) / rmses.length).toFixed(2) : "—";

  document.getElementById("sumMape").textContent = avgMape + "%";
  document.getElementById("sumRmse").textContent = avgRmse;
  document.getElementById("sumCount").textContent = commodities.length;

  const lastDate = commodities[0]?.fullHistory?.slice(-1)[0]?.date;
  document.getElementById("lastUpdated").textContent = lastDate
    ? `Updated ${formatShortDate(lastDate)}` : "Live";
}

// ============================================================
// TICKER MAPPING
// ============================================================
const TICKER_MAP = {
  "Gold": "XAU",
  "Silver": "XAG",
  "Crude oil": "CL",
  "Natural Gas": "NG",
  "Copper": "HG",
};

// ============================================================
// CARD GRID
// ============================================================
function renderGrid() {
  const grid = document.getElementById("cards-grid");
  grid.innerHTML = commodities.map((c, i) => cardHTML(c, i)).join("");

  commodities.forEach((c, i) => {
    drawSparkline(c, i);
    document.getElementById(`card-${i}`).addEventListener("click", () => selectCommodity(c));
  });
}

function cardHTML(c, i) {
  const isUp = c.changePct >= 0;
  const dirClass = isUp ? "up" : "down";
  const sign = isUp ? "+" : "";
  const ticker = TICKER_MAP[c.name] || c.name.slice(0, 3).toUpperCase();

  return `
    <div class="commodity-card" id="card-${i}" data-name="${c.safeName}">
      <!-- Currency pair header -->
      <div class="card-pair-header">
        <div class="card-pair-left">
          <span class="card-ticker">${ticker}</span>
          <span class="card-pair-label">${c.name}</span>
        </div>
        <span class="card-arrow">→</span>
        <div class="card-pair-right">
          <span class="card-flag-inr"><span class="flag-emoji">🇮🇳</span> INR ₹</span>
          <span class="card-pair-label">Indian Rupee FX</span>
        </div>
      </div>

      <!-- Price row -->
      <div class="card-price-row">
        <span class="card-price">${formatPrice(c.price)}</span>
        <span class="card-change ${dirClass}">${sign}${c.changePct.toFixed(2)}%</span>
      </div>

      <!-- Sparkline -->
      <div class="card-sparkline">
        <div class="card-prev-close">Previous close ${formatPrice(c.prevClose)}</div>
        <div class="card-spark-pct ${dirClass}">${sign}${c.changePct.toFixed(2)}%</div>
        <canvas id="spark-${i}"></canvas>
      </div>

      <!-- Timeframe pills -->
      <div class="card-timeframes">
        ${["1D","5D","1M","1Y","5Y","Max"].map((tf,j) =>
          `<button class="card-tf-pill ${j===0 ? "active" : ""}">${tf}</button>`).join("")}
      </div>

      <a class="card-see-more" href="javascript:void(0)">See full forecast →</a>
    </div>`;
}

function drawSparkline(c, i) {
  const canvas = document.getElementById(`spark-${i}`);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const isUp = c.changePct >= 0;
  const color = isUp ? "#3FB68B" : "#E85D5D";
  const bgColor = isUp ? "rgba(63,182,139,0.10)" : "rgba(232,93,93,0.10)";
  const n = c.sparkline.length;

  if (sparkCharts[i]) sparkCharts[i].destroy();
  sparkCharts[i] = new Chart(ctx, {
    type: "line",
    data: {
      labels: c.sparkline.map(p => p.date),
      datasets: [
        {
          label: "Price",
          data: c.sparkline.map(p => p.price),
          borderColor: color,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0.35,
          fill: true,
          backgroundColor: bgColor,
          order: 1,
        },
        {
          label: "Previous close",
          data: new Array(n).fill(c.prevClose),
          borderColor: "#5A5E66",
          borderWidth: 1,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
          order: 2,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
      animation: false,
    }
  });
}

// ============================================================
// SELECT COMMODITY → SHOW DETAIL BELOW CARDS
// ============================================================
function selectCommodity(c) {
  activeCommodity = c;
  activeTimeframe = "1M";

  // Highlight active card
  document.querySelectorAll(".commodity-card").forEach(el => el.classList.remove("active"));
  const idx = commodities.indexOf(c);
  const card = document.getElementById(`card-${idx}`);
  if (card) card.classList.add("active");

  renderDetailHeader(c);
  renderDetailMetrics(c);
  renderDetailChart(c);
  renderRecommendation(c);
  renderForecastTable(c);

  const section = document.getElementById("detail-section");
  section.classList.remove("hidden");
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderDetailHeader(c) {
  const isUp = c.changePct >= 0;
  document.getElementById("detailIcon").textContent = c.icon;
  document.getElementById("detailName").textContent = c.name;
  document.getElementById("detailUnit").textContent = c.unit;
  document.getElementById("detailPrice").textContent = "₹" + formatPrice(c.price);
  const changeEl = document.getElementById("detailChange");
  changeEl.textContent = `${isUp ? "▲" : "▼"} ${Math.abs(c.changePct).toFixed(2)}% today`;
  changeEl.className = "detail-change " + (isUp ? "up" : "down");
}

function renderDetailMetrics(c) {
  const m = c.metrics;
  const pills = [
    { label: "MAPE", value: m.mape ? m.mape.toFixed(2) + "%" : "—" },
    { label: "RMSE", value: m.rmse ? m.rmse.toFixed(2) : "—" },
    { label: "Direction Acc.", value: m.directional_accuracy ? m.directional_accuracy + "%" : "—" },
    { label: "LGBM Weight", value: m.w_lgbm ? (m.w_lgbm * 100).toFixed(0) + "%" : "—" },
    { label: "SARIMA Weight", value: m.w_sarima ? (m.w_sarima * 100).toFixed(0) + "%" : "—" },
  ];
  document.getElementById("detailMetrics").innerHTML = pills.map(p =>
    `<div class="metric-pill"><span class="metric-pill-label">${p.label}</span><span class="metric-pill-value">${p.value}</span></div>`
  ).join("");
}

function renderRecommendation(c) {
  const first = c.forecast[0];
  if (!first) return;
  const cls = first.recommendation.toLowerCase();
  document.getElementById("recommendBanner").innerHTML = `
    <span class="rec-badge ${cls}">${first.recommendation}</span>
    <span class="rec-reason">${first.reasoning}</span>
    <span class="rec-reason" style="margin-left:auto">Horizon: 1–30 days</span>
  `;
}

// ============================================================
// DETAIL CHART — history (blue) + forecast (green) + CI band + LGBM + SARIMA
// ============================================================
function renderDetailChart(c) {
  const ctx = document.getElementById("detailChart").getContext("2d");

  // Filter history by timeframe
  const history = filterByTimeframe(c.fullHistory, activeTimeframe);
  const forecast = c.forecast;

  // History data
  const histLabels = history.map(p => p.date);
  const histPrices = history.map(p => p.price);

  // Forecast data
  const fcstLabels = forecast.map(f => f.date);
  const fcstPrices = forecast.map(f => f.ensemble);
  const lgbmPrices = forecast.map(f => f.lgbm);
  const sarimaPrices = forecast.map(f => f.sarima);
  const ciHigh = forecast.map(f => f.ciHigh);
  const ciLow = forecast.map(f => f.ciLow);

  // Build combined labels
  const allLabels = [...histLabels, ...fcstLabels];

  // History series (null for forecast period)
  const histData = [...histPrices, ...new Array(fcstLabels.length).fill(null)];
  // Forecast series (null for history period)
  const fcstData = [...new Array(histLabels.length).fill(null), ...fcstPrices];
  // LGBM series
  const lgbmData = [...new Array(histLabels.length).fill(null), ...lgbmPrices];
  // SARIMA series
  const sarimaData = [...new Array(histLabels.length).fill(null), ...sarimaPrices];
  // CI high
  const ciHighData = [...new Array(histLabels.length).fill(null), ...ciHigh];
  // CI low
  const ciLowData = [...new Array(histLabels.length).fill(null), ...ciLow];

  // Bridge point: last history value carried into first forecast slot
  const bridgeIdx = histLabels.length;
  if (bridgeIdx > 0 && fcstPrices.length > 0) {
    fcstData[bridgeIdx - 1] = histPrices[histPrices.length - 1];
    lgbmData[bridgeIdx - 1] = histPrices[histPrices.length - 1];
    sarimaData[bridgeIdx - 1] = histPrices[histPrices.length - 1];
    ciHighData[bridgeIdx - 1] = histPrices[histPrices.length - 1];
    ciLowData[bridgeIdx - 1] = histPrices[histPrices.length - 1];
  }

  if (detailChart) detailChart.destroy();

  detailChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "CI High",
          data: ciHighData,
          borderColor: "transparent",
          backgroundColor: "rgba(34,197,94,0.10)",
          fill: "+1",
          pointRadius: 0,
          order: 5,
        },
        {
          label: "CI Low",
          data: ciLowData,
          borderColor: "transparent",
          backgroundColor: "transparent",
          fill: false,
          pointRadius: 0,
          order: 5,
        },
        {
          label: "Historical",
          data: histData,
          borderColor: "#3B82F6",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
          fill: false,
          order: 2,
        },
        {
          label: "Ensemble Forecast",
          data: fcstData,
          borderColor: "#22C55E",
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.2,
          fill: false,
          order: 1,
        },
        {
          label: "LightGBM",
          data: lgbmData,
          borderColor: "#F59E0B",
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.2,
          fill: false,
          order: 3,
        },
        {
          label: "SARIMA",
          data: sarimaData,
          borderColor: "#A78BFA",
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          tension: 0.2,
          fill: false,
          order: 4,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1A1D22",
          borderColor: "#24282F",
          borderWidth: 1,
          titleColor: "#E8E6E1",
          bodyColor: "#8B8F98",
          padding: 12,
          displayColors: true,
          callbacks: {
            title: (items) => items[0]?.label || "",
            label: (item) => {
              if (item.raw === null) return null;
              return ` ${item.dataset.label}: ₹${Number(item.raw).toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#5A5E66", maxTicksLimit: 10, font: { size: 10 } },
          grid: { color: "rgba(36,40,47,0.5)", drawBorder: false },
        },
        y: {
          ticks: {
            color: "#5A5E66",
            font: { size: 10 },
            callback: (v) => "₹" + formatCompact(v),
          },
          grid: { color: "rgba(36,40,47,0.5)", drawBorder: false },
        },
      },
    }
  });

  // Wire timeframe buttons
  document.querySelectorAll("#detailTimeframes .tf-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll("#detailTimeframes .tf-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeTimeframe = btn.dataset.tf;
      renderDetailChart(activeCommodity);
    };
  });
}

// ============================================================
// FORECAST TABLE
// ============================================================
function renderForecastTable(c) {
  const tbody = document.getElementById("forecastTableBody");
  tbody.innerHTML = c.forecast.map((f, i) => {
    const isUp = f.pctChange >= 0;
    const dirClass = isUp ? "up" : "down";
    const sign = isUp ? "+" : "";
    const sigClass = f.recommendation.toLowerCase();
    return `<tr>
      <td>${f.horizon}</td>
      <td>${formatShortDate(f.date)}</td>
      <td><strong>₹${formatPrice(f.ensemble)}</strong></td>
      <td>₹${formatPrice(f.lgbm)}</td>
      <td>₹${formatPrice(f.sarima)}</td>
      <td>₹${formatPrice(f.ciLow)}</td>
      <td>₹${formatPrice(f.ciHigh)}</td>
      <td class="${dirClass}" style="color:${isUp ? 'var(--green)' : 'var(--red)'}">${sign}${f.pctChange.toFixed(2)}%</td>
      <td><span class="table-signal signal-${sigClass}">${f.recommendation}</span></td>
    </tr>`;
  }).join("");
}

// ============================================================
// HELPERS
// ============================================================
function formatPrice(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function formatCompact(v) {
  if (Math.abs(v) >= 100000) return (v / 100000).toFixed(1) + "L";
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + "K";
  return v.toFixed(0);
}

function formatShortDate(dateStr) {
  try {
    const d = new Date(dateStr + "T00:00:00");
    return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "2-digit" });
  } catch { return dateStr; }
}

function filterByTimeframe(history, tf) {
  if (tf === "MAX" || !history.length) return history;
  const now = new Date(history[history.length - 1].date + "T00:00:00");
  const months = { "1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36 }[tf] || 1;
  const cutoff = new Date(now);
  cutoff.setMonth(cutoff.getMonth() - months);
  return history.filter(p => new Date(p.date + "T00:00:00") >= cutoff);
}

// ============================================================
// CLOSE DETAIL
// ============================================================
document.addEventListener("click", (e) => {
  if (e.target.id === "detailClose" || e.target.closest("#detailClose")) {
    document.getElementById("detail-section").classList.add("hidden");
    document.querySelectorAll(".commodity-card").forEach(el => el.classList.remove("active"));
    activeCommodity = null;
  }
});

// ============================================================
// INIT
// ============================================================
async function init() {
  const grid = document.getElementById("cards-grid");
  grid.innerHTML = `
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p class="loading-text">Loading forecast data...</p>
    </div>`;

  try {
    const raw = await loadForecasts();
    processData(raw);
    renderSummary();
    renderGrid();
  } catch (err) {
    grid.innerHTML = `
      <div class="loading-state">
        <p class="error-text">Failed to load forecasts: ${err.message}</p>
        <p class="loading-text" style="margin-top:8px">Run <code>python src/save_forecasts.py</code> to generate data.</p>
      </div>`;
  }
}

init();
