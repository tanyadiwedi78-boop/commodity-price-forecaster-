const API_URL = "http://localhost:8000";

let commodities = [];
let sparkCharts = {};
let fullChart = null;
let currentModalCommodity = null;

const ALL_TIMEFRAMES = ["1D", "5D", "1M", "3M", "YTD", "1Y", "3Y", "5Y", "Max"];
const DAY_COUNTS = { "1D": 2, "5D": 5, "1M": 30, "3M": 90, "1Y": 365, "3Y": 1095, "5Y": 1825 };

// ============================================================
// DATA LOADING
// ============================================================
async function loadForecasts() {
  const res = await fetch(`${API_URL}/forecasts`);
  if (!res.ok) throw new Error(`Failed to load forecasts: ${res.status}`);
  const raw = await res.json();

  return Object.values(raw).map((c) => {
    const historyEntries = Object.entries(c.history).sort(([a], [b]) => a.localeCompare(b));
    const last = historyEntries[historyEntries.length - 1];
    const prev = historyEntries[historyEntries.length - 2];

    const price = last ? last[1] : 0;
    const prevClose = prev ? prev[1] : price;
    const changePct = prevClose ? ((price - prevClose) / prevClose) * 100 : 0;

    const fullHistory = historyEntries.map(([date, p]) => ({ date, price: p }));
    const forecast = (c.forecast || []).map((f) => ({
      date: f.date, price: f.ensemble_pred, ci_low: f.ci_low, ci_high: f.ci_high,
    }));

    return {
      name: c.name, icon: c.icon, unit: c.unit,
      price, changePct, prevClose, fullHistory, forecast,
      _activeTf: "1M", _modalTf: "1M",
    };
  });
}

function filterHistory(fullHistory, tf) {
  if (tf === "Max") return fullHistory;
  if (tf === "YTD") {
    const currentYear = new Date().getFullYear();
    return fullHistory.filter(p => new Date(p.date).getFullYear() === currentYear);
  }
  const days = DAY_COUNTS[tf] || fullHistory.length;
  return fullHistory.slice(-days);
}

// ============================================================
// RENDER ALL 5 CARDS
// ============================================================
function renderGrid() {
  const grid = document.getElementById("cards-grid");
  grid.innerHTML = commodities.map((c, i) => cardHTML(c, i)).join("");

  commodities.forEach((c, i) => {
    drawSparkline(c, i);

    document.getElementById(`pairRow-${i}`).onclick = () => openModal(c);
    document.getElementById(`expandHint-${i}`).onclick = () => openModal(c);

    document.querySelectorAll(`#tfRow-${i} .tf-pill`).forEach((btn) => {
      btn.onclick = () => {
        document.querySelectorAll(`#tfRow-${i} .tf-pill`).forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        c._activeTf = btn.dataset.tf;
        drawSparkline(c, i);
      };
    });
  });
}

function cardHTML(c, i) {
  const isUp = c.changePct >= 0;
  const colorClass = isUp ? "up" : "down";
  const sign = isUp ? "+" : "";

  return `
    <div class="widget-card">
      <div class="widget-header">
        <div><span class="up-arrow">↗</span> ${c.name}</div>
        <span class="menu-dots">•••</span>
      </div>

      <div class="pair-row" id="pairRow-${i}">
        <div>
          <div class="pair-name">${c.icon} ${c.name}</div>
          <div class="pair-sub">${c.unit}</div>
        </div>
        <span class="pair-arrow">→</span>
        <div>
          <div class="pair-name">🇮🇳 INR ₹</div>
          <div class="pair-sub">Indian Rupee</div>
        </div>
      </div>

      <div class="price-row">
        <span class="price-big">${c.price.toLocaleString("en-IN")}</span>
        <span class="price-change ${colorClass}">${sign}${c.changePct.toFixed(2)}%</span>
      </div>

      <div class="sparkline-wrap">
        <div class="prev-close-label">Prev close ${c.prevClose.toLocaleString("en-IN")}</div>
        <canvas id="sparkCanvas-${i}"></canvas>
        <div class="spark-pct-label ${colorClass}">${sign}${c.changePct.toFixed(2)}%</div>
      </div>

      <div class="timeframe-row" id="tfRow-${i}">
        ${ALL_TIMEFRAMES.map((tf) =>
          `<button class="tf-pill ${tf==="1M" ? "active" : ""}" data-tf="${tf}">${tf}</button>`).join("")}
      </div>

      <div class="widget-footer">
        <span class="expand-hint" id="expandHint-${i}">Expand chart to see the full forecast →</span>
      </div>
    </div>
  `;
}

function drawSparkline(c, i) {
  const ctx = document.getElementById(`sparkCanvas-${i}`).getContext("2d");
  const isUp = c.changePct >= 0;
  const lineColor = isUp ? "#3FB68B" : "#E85D5D";
  const filtered = filterHistory(c.fullHistory, c._activeTf);

  if (sparkCharts[i]) sparkCharts[i].destroy();
  sparkCharts[i] = new Chart(ctx, {
    type: "line",
    data: {
      labels: filtered.map(p => p.date),
      datasets: [{
        data: filtered.map(p => p.price),
        borderColor: lineColor,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
    }
  });
}

// ============================================================
// EXPANDED MODAL
// ============================================================
function openModal(c) {
  currentModalCommodity = c;
  currentModalCommodity._modalTf = "1M";

  const isUp = c.changePct >= 0;
  const colorClass = isUp ? "up" : "down";
  const sign = isUp ? "▲" : "▼";

  document.getElementById("modal-content").innerHTML = `
    <div class="modal-header">
      <div>
        <div style="font-size:14px;color:#8B8F98">${c.icon} ${c.name}</div>
        <div class="modal-price">₹${c.price.toLocaleString("en-IN")}</div>
        <div class="modal-change ${colorClass}">${sign} ${Math.abs(c.changePct).toFixed(2)}%</div>
      </div>
      <button class="modal-close" id="closeModalBtn">✕</button>
    </div>

    <div class="modal-timeframes" id="modalTfRow">
      ${ALL_TIMEFRAMES.map((tf) =>
        `<button class="modal-tf-pill ${tf==="1M" ? "active" : ""}" data-tf="${tf}">${tf}</button>`).join("")}
    </div>

    <div id="full-chart-canvas-wrap"><canvas id="fullChartCanvas"></canvas></div>

    <div class="quick-compare" id="quickCompare"></div>
  `;

  document.getElementById("modal-overlay").classList.remove("hidden");
  document.getElementById("closeModalBtn").onclick = closeModal;

  document.querySelectorAll("#modalTfRow .modal-tf-pill").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#modalTfRow .modal-tf-pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentModalCommodity._modalTf = btn.dataset.tf;
      drawFullChart(currentModalCommodity);
    };
  });

  drawFullChart(c);
  renderQuickCompare(c);
}

function drawFullChart(c) {
  const ctx = document.getElementById("fullChartCanvas").getContext("2d");
  const isUp = c.changePct >= 0;
  const lineColor = isUp ? "#3FB68B" : "#E85D5D";

  const filteredHistory = filterHistory(c.fullHistory, c._modalTf || "1M");
  const combined = [...filteredHistory, ...c.forecast];

  const labels = combined.map(p => p.date);
  const prices = combined.map(p => p.price);
  const ciHigh = combined.map(p => p.ci_high ?? null);
  const ciLow = combined.map(p => p.ci_low ?? null);

  if (fullChart) fullChart.destroy();
  fullChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Confidence High", data: ciHigh, borderColor: "transparent",
          backgroundColor: "rgba(201,161,90,0.12)", fill: "+1", pointRadius: 0 },
        { label: "Confidence Low", data: ciLow, borderColor: "transparent",
          backgroundColor: "rgba(201,161,90,0.12)", fill: false, pointRadius: 0 },
        { label: "Price", data: prices, borderColor: lineColor, borderWidth: 2,
          pointRadius: 0, tension: 0.2, fill: false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8B8F98", maxTicksLimit: 8 }, grid: { color: "#24282F" } },
        y: { ticks: { color: "#8B8F98" }, grid: { color: "#24282F" } },
      },
    }
  });
}

function renderQuickCompare(current) {
  const others = commodities.filter(c => c.name !== current.name).slice(0, 2);
  document.getElementById("quickCompare").innerHTML = others.map(c => {
    const isUp = c.changePct >= 0;
    return `
      <div class="compare-box">
        <div class="compare-name">${c.icon} ${c.name}</div>
        <div class="compare-price ${isUp ? 'up' : 'down'}">
          ₹${c.price.toLocaleString("en-IN")} ${isUp ? '+' : ''}${c.changePct.toFixed(2)}%
        </div>
      </div>
    `;
  }).join("");
}

function closeModal() {
  document.getElementById("modal-overlay").classList.add("hidden");
}
document.getElementById("modal-overlay").onclick = (e) => {
  if (e.target.id === "modal-overlay") closeModal();
};

// ============================================================
// INIT
// ============================================================
loadForecasts()
  .then((data) => { commodities = data; renderGrid(); })
  .catch((err) => {
    document.getElementById("cards-grid").innerHTML =
      `<p class="error-msg">Error loading forecasts: ${err.message}</p>`;
  });