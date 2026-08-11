// Configurable backend URL -- swap the production URL once FastAPI is deployed
const API_BASE_URL = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
    ? "http://127.0.0.1:8000"
    : "https://your-fastapi-app.onrender.com";   // TODO: update once deployed

const STATIC_DATA_URL = "data/forecasts.json";

const COMMODITIES = {
    gold:        { name: "Gold",         icon: "🥇", unit: "₹/10g" },
    silver:      { name: "Silver",       icon: "🥈", unit: "₹/kg" },
    crude_oil:   { name: "Crude Oil",    icon: "🛢",  unit: "₹/barrel" },
    natural_gas: { name: "Natural Gas",  icon: "🔥", unit: "₹/MMBtu" },
    copper:      { name: "Copper",       icon: "🟤", unit: "₹/kg" },
};

// Generates ~5 years of synthetic daily history for mock/testing mode only.
// Real data (from export_forecasts.py) already includes full history.
function generateMockHistory(basePrice, volatility) {
    const history = {};
    let price = basePrice * 0.7;   // start lower, trend up over 5 years
    const days = 1825;
    const start = new Date();
    start.setDate(start.getDate() - days);

    for (let i = 0; i < days; i++) {
        const date = new Date(start);
        date.setDate(date.getDate() + i);
        price += (Math.random() - 0.48) * volatility;   // slight upward drift
        price = Math.max(price, basePrice * 0.3);
        history[date.toISOString().split("T")[0]] = Math.round(price * 100) / 100;
    }
    return history;
}

const MOCK_HISTORY = {
    gold: generateMockHistory(73400, 400),
    silver: generateMockHistory(92100, 500),
    crude_oil: generateMockHistory(7020, 60),
    natural_gas: generateMockHistory(221, 4),
    copper: generateMockHistory(833, 8),
};

// --- MOCK DATA -- lets you build and test the whole UI before the
// FastAPI backend exists. Swap 'usingMock' logic once real data flows. ---
const MOCK_DATA = {
    gold: {
        name: "Gold", icon: "🥇", unit: "₹/10g",
        history: MOCK_HISTORY.gold,
        forecast: [
            { date: "2026-07-23", horizon_days: 1, ensemble_pred: 73520, lgbm_pred: 73610, sarima_pred: 73400, recommendation: "BUY", reasoning: "Predicted +0.16% exceeds the model's typical error margin (0.90%)", pct_change: 0.16 },
            { date: "2026-07-24", horizon_days: 2, ensemble_pred: 73580 },
            { date: "2026-07-25", horizon_days: 3, ensemble_pred: 73650 },
        ],
        metrics: { rmse: 54.30, mape: 0.9, w_lgbm: 0.55, w_sarima: 0.45 },
    },
    silver: {
        name: "Silver", icon: "🥈", unit: "₹/kg",
        history: MOCK_HISTORY.silver,
        forecast: [
            { date: "2026-07-23", horizon_days: 1, ensemble_pred: 92300, lgbm_pred: 92450, sarima_pred: 92100, recommendation: "HOLD", reasoning: "Predicted +0.22% is within the model's noise range (±1.40%) -- not a confident signal", pct_change: 0.22 },
            { date: "2026-07-24", horizon_days: 2, ensemble_pred: 92450 },
            { date: "2026-07-25", horizon_days: 3, ensemble_pred: 92600 },
        ],
        metrics: { rmse: 210.5, mape: 1.4, w_lgbm: 0.55, w_sarima: 0.45 },
    },
    crude_oil: {
        name: "Crude Oil", icon: "🛢", unit: "₹/barrel",
        history: MOCK_HISTORY.crude_oil,
        forecast: [
            { date: "2026-07-23", horizon_days: 1, ensemble_pred: 7040, lgbm_pred: 7080, sarima_pred: 6990, recommendation: "SELL", reasoning: "Predicted -1.85% falls below the model's typical error margin (-1.80%)", pct_change: -1.85 },
            { date: "2026-07-24", horizon_days: 2, ensemble_pred: 7010 },
            { date: "2026-07-25", horizon_days: 3, ensemble_pred: 6980 },
        ],
        metrics: { rmse: 78.1, mape: 1.8, w_lgbm: 0.55, w_sarima: 0.45 },
    },
    natural_gas: {
        name: "Natural Gas", icon: "🔥", unit: "₹/MMBtu",
        history: MOCK_HISTORY.natural_gas,
        forecast: [
            { date: "2026-07-23", horizon_days: 1, ensemble_pred: 222, lgbm_pred: 224, sarima_pred: 219 },
            { date: "2026-07-24", horizon_days: 2, ensemble_pred: 223 },
            { date: "2026-07-25", horizon_days: 3, ensemble_pred: 225 },
        ],
        metrics: { rmse: 4.2, mape: 2.1, w_lgbm: 0.55, w_sarima: 0.45 },
    },
    copper: {
        name: "Copper", icon: "🟤", unit: "₹/kg",
        history: MOCK_HISTORY.copper,
        forecast: [
            { date: "2026-07-23", horizon_days: 1, ensemble_pred: 835, lgbm_pred: 837, sarima_pred: 832 },
            { date: "2026-07-24", horizon_days: 2, ensemble_pred: 837 },
            { date: "2026-07-25", horizon_days: 3, ensemble_pred: 839 },
        ],
        metrics: { rmse: 6.7, mape: 1.1, w_lgbm: 0.55, w_sarima: 0.45 },
    },
};

let cachedApiData = null;
let cachedStaticData = null;
let usingMock = false;
let chartInstance = null;

// --- DATA FETCHING: live API -> static JSON -> mock, in that order ---
async function loadAllData() {
    // 1. Try the live FastAPI backend
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);
        const res = await fetch(`${API_BASE_URL}/api/dashboard`, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error(`API status ${res.status}`);
        cachedApiData = await res.json();
        setStatus("live", "Live API connected");
        return cachedApiData;
    } catch (err) {
        console.warn("Live API unreachable, falling back to static JSON:", err.message);
    }

    // 2. Try the static forecasts.json exported by export_forecasts.py
    try {
        const res = await fetch(STATIC_DATA_URL);
        if (!res.ok) throw new Error("forecasts.json not found");
        cachedStaticData = await res.json();
        setStatus("live", "Static forecast data loaded");
        return cachedStaticData;
    } catch (err) {
        console.warn("Static JSON unreachable, falling back to mock data:", err.message);
    }

    // 3. Fall back to mock data -- lets the UI work standalone
    usingMock = true;
    setStatus("mock", "Showing mock data (run save_forecasts.py + export_forecasts.py for live data)");
    return MOCK_DATA;
}

function setStatus(kind, text) {
    const dot = document.getElementById("status-dot");
    const label = document.getElementById("status-text");
    dot.className = "pulse-dot " + kind;
    label.textContent = text;
}

// --- TICKER STRIP ---
function renderTicker(data) {
    const track = document.getElementById("ticker-track");
    const items = Object.keys(COMMODITIES).map((key) => {
        const c = data[key];
        if (!c) return "";
        const histDates = Object.keys(c.history);
        const lastPrice = c.history[histDates[histDates.length - 1]];
        const nextForecast = c.forecast[0] ? c.forecast[0].ensemble_pred : lastPrice;
        const change = ((nextForecast - lastPrice) / lastPrice) * 100;
        const cls = change >= 0 ? "price-up" : "price-down";
        const arrow = change >= 0 ? "▲" : "▼";
        return `<span class="ticker-item">${c.icon} ${c.name}: ₹${lastPrice.toLocaleString("en-IN")}
                <span class="${cls}">${arrow} ${Math.abs(change).toFixed(2)}%</span></span>`;
    });
    // Duplicate the list so the CSS scroll animation loops seamlessly
    track.innerHTML = items.join("") + items.join("");
}

// --- CARDS ---
function renderCards(data) {
    const grid = document.getElementById("cards-grid");
    grid.innerHTML = "";

    Object.keys(COMMODITIES).forEach((key) => {
        const c = data[key];
        if (!c) return;

        const histDates = Object.keys(c.history);
        const lastPrice = c.history[histDates[histDates.length - 1]];
        const nextForecast = c.forecast[0] ? c.forecast[0].ensemble_pred : lastPrice;
        const change = ((nextForecast - lastPrice) / lastPrice) * 100;
        const trendClass = change >= 0 ? "up" : "down";
        const arrow = change >= 0 ? "↑" : "↓";

        const rec = c.forecast[0] ? c.forecast[0].recommendation : null;
        const recClass = rec ? rec.toLowerCase() : "hold";

        const card = document.createElement("div");
        card.className = "commodity-card";
        card.tabIndex = 0;
        card.innerHTML = `
            <div class="card-top">
                <span class="card-icon">${c.icon}</span>
                <span class="card-name">${c.name}</span>
            </div>
            ${rec ? `<span class="rec-badge ${recClass}">${rec}</span>` : ""}
            <div class="card-price">₹${nextForecast.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
            <div class="card-trend ${trendClass}">${arrow} ${Math.abs(change).toFixed(2)}% vs last close</div>
            <div class="card-footer">
                <span>${c.unit}</span>
                <span>RMSE: ${c.metrics.rmse ?? "—"}</span>
            </div>
        `;
        card.addEventListener("click", () => expandChart(key, c));
        card.addEventListener("keydown", (e) => { if (e.key === "Enter") expandChart(key, c); });
        grid.appendChild(card);
    });
}

// --- CHART ---
let currentCommodity = null;   // remembers which commodity is open, for timeframe switching

function expandChart(key, c) {
    currentCommodity = c;
    const section = document.getElementById("chart-section");
    section.classList.remove("hidden");
    section.scrollIntoView({ behavior: "smooth", block: "start" });

    document.getElementById("chart-title").textContent = `${c.icon} ${c.name} — Price Forecast`;
    document.getElementById("meta-rmse").textContent = c.metrics.rmse ? c.metrics.rmse.toFixed(2) : "—";
    document.getElementById("meta-split").textContent =
        c.metrics.w_lgbm ? `LightGBM ${Math.round(c.metrics.w_lgbm * 100)}% · SARIMA ${Math.round(c.metrics.w_sarima * 100)}%` : "—";

    const day1 = c.forecast[0];
    const recBox = document.getElementById("recommendation-box");
    if (day1 && day1.recommendation) {
        recBox.classList.remove("hidden");
        const badge = document.getElementById("rec-badge-large");
        badge.textContent = day1.recommendation;
        badge.className = `rec-badge ${day1.recommendation.toLowerCase()}`;
        document.getElementById("rec-reasoning").textContent = day1.reasoning || "";
    } else {
        recBox.classList.add("hidden");
    }

    // Default to 1Y view when opening a new commodity
    document.querySelectorAll(".tf-btn").forEach((b) => b.classList.remove("active"));
    document.querySelector('.tf-btn[data-days="365"]').classList.add("active");

    updatePriceHeader(c, 365);
    drawChart(c, 365);
}

function updatePriceHeader(c, days) {
    const histDates = Object.keys(c.history).sort();
    const slice = histDates.slice(-days);
    if (slice.length === 0) return;

    const latest = c.history[slice[slice.length - 1]];
    const earliest = c.history[slice[0]];
    const change = ((latest - earliest) / earliest) * 100;
    const trendClass = change >= 0 ? "up" : "down";
    const arrow = change >= 0 ? "▲" : "▼";

    document.getElementById("price-big").textContent = `₹${latest.toLocaleString("en-IN")}`;
    const changeEl = document.getElementById("price-change");
    changeEl.textContent = `${arrow} ${Math.abs(change).toFixed(2)}%`;
    changeEl.className = `price-change ${trendClass}`;
}

function drawChart(c, timeframeDays) {
    const ctx = document.getElementById("priceChart").getContext("2d");

    const allHistDates = Object.keys(c.history).sort();
    const histDates = timeframeDays >= 99999 ? allHistDates : allHistDates.slice(-timeframeDays);
    const histValues = histDates.map((d) => c.history[d]);

    const fcDates = c.forecast.map((d) => d.date);
    const fcValues = c.forecast.map((d) => d.ensemble_pred);

    const allLabels = [...histDates, ...fcDates];
    const historyData = [...histValues, ...Array(fcDates.length).fill(null)];
    const lastHistValue = histValues[histValues.length - 1];
    const forecastData = [...Array(histDates.length - 1).fill(null), lastHistValue, ...fcValues];

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: "Historical",
                    data: historyData,
                    borderColor: "#8B8F98",
                    backgroundColor: "rgba(139, 143, 152, 0.08)",
                    borderWidth: 2,
                    pointRadius: 2,
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: "Forecast",
                    data: forecastData,
                    borderColor: "#C9A15A",
                    borderWidth: 2.5,
                    borderDash: [6, 4],
                    pointRadius: 4,
                    pointBackgroundColor: "#C9A15A",
                    fill: false,
                    tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#8B8F98", font: { family: "JetBrains Mono", size: 11 } } },
                tooltip: {
                    backgroundColor: "#14171C",
                    titleColor: "#E8E6E1",
                    bodyColor: "#C9A15A",
                    borderColor: "#24282F",
                    borderWidth: 1,
                    padding: 10,
                },
            },
            scales: {
                x: { grid: { color: "#1A1E24" }, ticks: { color: "#8B8F98", maxTicksLimit: 10 } },
                y: { grid: { color: "#1A1E24" }, ticks: { color: "#8B8F98" } },
            },
        },
    });
}

// --- PARTICLE BACKGROUND ---
function initParticles() {
    const canvas = document.getElementById("particles-bg");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    const particles = Array.from({ length: 45 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 1.8 + 0.5,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
        alpha: Math.random() * 0.4 + 0.1,
    }));

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach((p) => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(201, 161, 90, ${p.alpha})`;
            ctx.fill();
        });
        requestAnimationFrame(animate);
    }

    // Respect reduced-motion preference
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        animate();
    }
}

// --- INIT ---
document.addEventListener("DOMContentLoaded", async () => {
    initParticles();

    document.getElementById("chart-close").addEventListener("click", () => {
        document.getElementById("chart-section").classList.add("hidden");
    });

    document.querySelectorAll(".tf-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            if (!currentCommodity) return;
            document.querySelectorAll(".tf-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const days = parseInt(btn.dataset.days, 10);
            updatePriceHeader(currentCommodity, days);
            drawChart(currentCommodity, days);
        });
    });

    const data = await loadAllData();
    renderTicker(data);
    renderCards(data);
});