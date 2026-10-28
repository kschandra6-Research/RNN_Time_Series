/**
 * NeuroCast: Frontend Client Application
 * Concept & Design by K.S.Chandra 2026
 * 
 * Provides an interactive, responsive user interface.
 * Connects to the secure backend REST API while keeping neural network architecture
 * and training routines hidden on the server.
 */

// Global Application State
const state = {
  series: [],
  predictions: [],
  lossHistory: [],
  lastResult: null,
  activeSource: "preset",
  activeTab: "forecast",
  availableColumns: {},
  sourceDescription: "Preset: sine_trend"
};

// DOM Element References
const dom = {
  engineChip: document.getElementById("engine-chip"),
  engineName: document.getElementById("engine-name"),
  
  // Segmented Source Buttons
  segPreset: document.getElementById("seg-preset"),
  segFile: document.getElementById("seg-file"),
  segManual: document.getElementById("seg-manual"),
  
  subviewPreset: document.getElementById("subview-preset"),
  subviewFile: document.getElementById("subview-file"),
  subviewManual: document.getElementById("subview-manual"),
  
  presetSelect: document.getElementById("preset-select"),
  fileDropzone: document.getElementById("file-dropzone"),
  fileInput: document.getElementById("file-input"),
  fileColumnContainer: document.getElementById("file-column-container"),
  columnSelect: document.getElementById("column-select"),
  manualInput: document.getElementById("manual-input"),
  btnApplyManual: document.getElementById("btn-apply-manual"),
  
  // Stats Box
  statSource: document.getElementById("stat-source"),
  statCount: document.getElementById("stat-count"),
  statRange: document.getElementById("stat-range"),
  statLast: document.getElementById("stat-last"),
  
  // Hyperparameters
  inputEpochs: document.getElementById("input-epochs"),
  inputLr: document.getElementById("input-lr"),
  selectHidden: document.getElementById("select-hidden"),
  selectHorizon: document.getElementById("select-horizon"),
  selectEngine: document.getElementById("select-engine"),
  
  // Action Buttons
  btnTrain: document.getElementById("btn-train"),
  btnReset: document.getElementById("btn-reset"),
  
  // Progress & Status
  statusIndicator: document.getElementById("status-indicator"),
  statusTitle: document.getElementById("status-title"),
  statusMetrics: document.getElementById("status-metrics"),
  progressBar: document.getElementById("progress-bar"),
  
  // Tabs & Canvases
  tabBtns: document.querySelectorAll(".tab-btn"),
  tabContents: {
    forecast: document.getElementById("tab-forecast-content"),
    loss: document.getElementById("tab-loss-content"),
    data: document.getElementById("tab-data-content")
  },
  forecastCanvas: document.getElementById("forecastCanvas"),
  lossCanvas: document.getElementById("lossCanvas"),
  chartTooltip: document.getElementById("chart-tooltip"),
  serverLog: document.getElementById("server-log"),
  
  // Metric Tiles
  resLast: document.getElementById("res-last"),
  resPred: document.getElementById("res-pred"),
  resDiff: document.getElementById("res-diff"),
  resLoss: document.getElementById("res-loss"),
  
  // Bottom Actions
  btnCopyReport: document.getElementById("btn-copy-report"),
  btnExportCsv: document.getElementById("btn-export-csv")
};

// ==============================================================================
// Synthetic Wave Presets Generator
// ==============================================================================
function generatePreset(type, n = 50) {
  const series = [];
  if (type === "sine_trend") {
    for (let i = 0; i < n; i++) {
      series.push(Number((Math.sin(i * 0.25) * 4.0 + (i * 0.4) + 12.0).toFixed(2)));
    }
  } else if (type === "stock_walk") {
    let p = 100.0;
    series.push(p);
    for (let i = 1; i < n; i++) {
      // Deterministic random walk for repeatable clean visualization
      const shock = Math.sin(i * 1.3) * 1.5 + (Math.cos(i * 0.7) * 0.8) + 0.2;
      p = Math.max(10.0, p + shock);
      series.push(Number(p.toFixed(2)));
    }
  } else if (type === "seasonal_sales") {
    for (let i = 0; i < n; i++) {
      series.push(Number((15.0 + 3.0 * Math.sin(i * 0.6) + 6.0 * Math.cos(i * 0.15) + i * 0.2).toFixed(2)));
    }
  } else if (type === "damped_oscillation") {
    for (let i = 0; i < n; i++) {
      series.push(Number((20.0 + 15.0 * Math.exp(-i * 0.05) * Math.cos(i * 0.4)).toFixed(2)));
    }
  }
  return series;
}

// ==============================================================================
// Initialization & Backend Health Probing
// ==============================================================================
async function initApp() {
  bindEvents();
  setupCanvasResize();
  
  // Probe Backend Health
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      dom.engineName.textContent = `Backend Online • ${data.engine_detected}`;
    } else {
      dom.engineName.textContent = "Backend: Offline";
      dom.engineChip.style.color = "#f43f5e";
    }
  } catch (e) {
    dom.engineName.textContent = "Backend: Local Mode";
  }

  // Load Initial Preset
  loadPresetData();
}

// ==============================================================================
// Event Listeners
// ==============================================================================
function bindEvents() {
  // Segmented Source Buttons (Eliminates white-on-white text hiding)
  dom.segPreset.addEventListener("click", () => switchSource("preset"));
  dom.segFile.addEventListener("click", () => switchSource("file"));
  dom.segManual.addEventListener("click", () => switchSource("manual"));

  dom.presetSelect.addEventListener("change", loadPresetData);

  // File Upload Handlers
  dom.fileDropzone.addEventListener("click", () => dom.fileInput.click());
  dom.fileInput.addEventListener("change", handleFileSelected);

  dom.fileDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dom.fileDropzone.classList.add("dragover");
  });
  dom.fileDropzone.addEventListener("dragleave", () => {
    dom.fileDropzone.classList.remove("dragover");
  });
  dom.fileDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dom.fileDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      dom.fileInput.files = e.dataTransfer.files;
      handleFileSelected();
    }
  });

  dom.columnSelect.addEventListener("change", () => {
    const col = dom.columnSelect.value;
    if (state.availableColumns && state.availableColumns[col]) {
      setSeriesData(state.availableColumns[col], `File Column: ${col}`);
    }
  });

  dom.btnApplyManual.addEventListener("click", handleManualInput);

  // Training & Reset
  dom.btnTrain.addEventListener("click", startTrainingWorkflow);
  dom.btnReset.addEventListener("click", resetDefaults);

  // Tabs
  dom.tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      dom.tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const tabName = btn.dataset.tab;
      state.activeTab = tabName;
      
      Object.keys(dom.tabContents).forEach(k => {
        dom.tabContents[k].classList.toggle("hidden", k !== tabName);
      });

      if (tabName === "forecast") drawForecastChart();
      if (tabName === "loss") drawLossChart();
    });
  });

  // Canvas Mouse Move Tooltips
  dom.forecastCanvas.addEventListener("mousemove", onForecastCanvasHover);
  dom.forecastCanvas.addEventListener("mouseleave", onForecastCanvasLeave);

  // Export Buttons
  dom.btnCopyReport.addEventListener("click", copyReportToClipboard);
  dom.btnExportCsv.addEventListener("click", exportForecastCsv);
}

// Quick Pill Button Helper
window.setEpochs = function(val) {
  dom.inputEpochs.value = val;
  document.querySelectorAll(".pill-btn").forEach(p => {
    p.classList.toggle("active", Number(p.textContent.trim()) === val);
  });
};

// ==============================================================================
// Source Switcher & Data Handling
// ==============================================================================
function switchSource(sourceKey) {
  state.activeSource = sourceKey;
  
  [dom.segPreset, dom.segFile, dom.segManual].forEach(btn => {
    btn.classList.toggle("active", btn.dataset.source === sourceKey);
  });

  dom.subviewPreset.classList.toggle("hidden", sourceKey !== "preset");
  dom.subviewFile.classList.toggle("hidden", sourceKey !== "file");
  dom.subviewManual.classList.toggle("hidden", sourceKey !== "manual");

  if (sourceKey === "preset") {
    loadPresetData();
  } else if (sourceKey === "manual") {
    handleManualInput();
  }
}

function loadPresetData() {
  const pType = dom.presetSelect.value;
  const data = generatePreset(pType, 50);
  setSeriesData(data, `Preset: ${pType}`);
}

async function handleFileSelected() {
  const file = dom.fileInput.files[0];
  if (!file) return;

  const text = await file.text();
  try {
    const res = await fetch("/api/inspect-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: text, filename: file.name })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "File parsing failed");
    }

    const data = await res.json();
    state.availableColumns = data.data;

    const colNames = Object.keys(data.columns);
    dom.columnSelect.innerHTML = "";
    colNames.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = `${c} (${data.columns[c].count} observations)`;
      dom.columnSelect.appendChild(opt);
    });

    dom.fileColumnContainer.classList.remove("hidden");
    const initialCol = colNames[0];
    setSeriesData(state.availableColumns[initialCol], `${file.name} [${initialCol}]`);

  } catch (err) {
    alert(`Error loading file: ${err.message}`);
  }
}

function handleManualInput() {
  const raw = dom.manualInput.value;
  const nums = [];
  raw.replace(/,/g, " ").split(/\s+/).forEach(t => {
    const v = parseFloat(t);
    if (!isNaN(v)) nums.push(v);
  });

  if (nums.length < 2) {
    alert("Please enter at least 2 valid numeric observations.");
    return;
  }
  setSeriesData(nums, "Manual Input");
}

function setSeriesData(series, sourceDesc) {
  state.series = series;
  state.predictions = [];
  state.sourceDescription = sourceDesc;
  
  const n = series.length;
  const minVal = Math.min(...series);
  const maxVal = Math.max(...series);
  const meanVal = series.reduce((a, b) => a + b, 0) / n;
  const lastVal = series[n - 1];

  dom.statSource.textContent = sourceDesc;
  dom.statCount.textContent = `${n} observations`;
  dom.statRange.textContent = `Min: ${minVal.toFixed(2)} | Max: ${maxVal.toFixed(2)}`;
  dom.statLast.textContent = lastVal.toFixed(4);
  dom.resLast.textContent = lastVal.toFixed(4);

  drawForecastChart();
}

// ==============================================================================
// Training Workflow & API Execution
// ==============================================================================
async function startTrainingWorkflow() {
  if (!state.series || state.series.length < 2) {
    alert("Please load at least 2 time-series observations.");
    return;
  }

  const epochs = parseInt(dom.inputEpochs.value, 10) || 1000;
  const lr = parseFloat(dom.inputLr.value) || 0.01;
  const hiddenSize = parseInt(dom.selectHidden.value, 10) || 32;
  const horizon = parseInt(dom.selectHorizon.value, 10) || 1;
  const engine = dom.selectEngine.value;

  dom.btnTrain.disabled = true;
  dom.statusIndicator.className = "status-dot training";
  dom.statusTitle.textContent = `Training RNN across ${epochs} epochs...`;
  dom.progressBar.style.width = "0%";

  logMessage("=== NeuroCast Training Request Dispatched ===");
  logMessage(`Target Observations: ${state.series.length}`);
  logMessage(`Epochs: ${epochs} (Default) | LR: ${lr} | Hidden: ${hiddenSize} | Horizon: ${horizon}`);
  logMessage("Concept & Design by K.S.Chandra 2026");
  logMessage("Communicating with hidden backend neural network...");

  // Simulated smooth progress while waiting for high-speed backend execution
  let simProgress = 0;
  const simTimer = setInterval(() => {
    if (simProgress < 90) {
      simProgress += 10;
      dom.progressBar.style.width = `${simProgress}%`;
      dom.statusMetrics.textContent = `Epoch ${Math.round((simProgress / 100) * epochs)} / ${epochs} | Optimizing...`;
    }
  }, 120);

  const tStart = performance.now();

  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        series: state.series,
        epochs: epochs,
        lr: lr,
        hidden_size: hiddenSize,
        forecast_steps: horizon,
        engine: engine
      })
    });

    clearInterval(simTimer);
    dom.progressBar.style.width = "100%";

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || "Training failed on server");
    }

    const result = await response.json();
    const duration = ((performance.now() - tStart) / 1000).toFixed(2);

    state.lastResult = result;
    state.predictions = result.predictions;
    state.lossHistory = result.loss_history;

    // Update Status
    dom.statusIndicator.className = "status-dot ready";
    dom.statusTitle.textContent = `Training Complete (${result.epochs_completed} epochs)`;
    dom.statusMetrics.textContent = `Epoch ${result.epochs_completed}/${epochs} (100%) | Loss: ${result.final_loss} | Time: ${duration}s`;

    // Update Metric Tiles
    dom.resLast.textContent = result.last_val.toFixed(4);
    dom.resPred.textContent = result.predicted_val.toFixed(4);

    const isPos = result.diff >= 0;
    dom.resDiff.textContent = `${result.direction_symbol} ${result.direction} of ${Math.abs(result.diff).toFixed(4)} (${result.pct_change >= 0 ? "+" : ""}${result.pct_change.toFixed(2)}%)`;
    dom.resDiff.className = `tile-value ${isPos ? "delta-pos" : "delta-neg"}`;
    dom.resLoss.textContent = result.final_loss;

    logMessage("-" * 45);
    logMessage(`Engine Used          : ${result.engine_used}`);
    logMessage(`Final MSE Loss       : ${result.final_loss}`);
    logMessage(`Last Observed Value  : ${result.last_val}`);
    logMessage(`Forecast Next Value  : ${result.predicted_val}`);
    logMessage(`Projected Delta      : ${result.direction} (${result.pct_change}%)`);
    if (result.predictions.length > 1) {
      logMessage(`Full Rollout Steps   : [${result.predictions.join(", ")}]`);
    }
    logMessage(`Backend Time         : ${result.training_time}s`);
    logMessage("=== Execution Successful ===");

    // Redraw charts
    drawForecastChart();
    drawLossChart();

  } catch (error) {
    clearInterval(simTimer);
    dom.statusIndicator.className = "status-dot";
    dom.statusTitle.textContent = "Training Failed";
    dom.statusMetrics.textContent = error.message;
    logMessage(`[ERROR] ${error.message}`);
    alert(`Training Error: ${error.message}`);
  } finally {
    dom.btnTrain.disabled = false;
  }
}

function resetDefaults() {
  dom.inputEpochs.value = 1000;
  setEpochs(1000);
  dom.inputLr.value = 0.01;
  dom.selectHidden.value = 32;
  dom.selectHorizon.value = 1;
  dom.selectEngine.value = "auto";
  dom.progressBar.style.width = "0%";
  dom.statusTitle.textContent = "Settings reset to defaults (1000 epochs).";
  dom.statusMetrics.textContent = "Epoch 0 / 1000 | Loss: -";
}

function logMessage(msg) {
  dom.serverLog.textContent += `\n${msg}`;
  const container = document.querySelector(".log-container");
  if (container) container.scrollTop = container.scrollHeight;
}

// ==============================================================================
// Interactive Canvas Chart Engine (Strict Hover Isolation)
// ==============================================================================
function setupCanvasResize() {
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas(dom.forecastCanvas);
    resizeCanvas(dom.lossCanvas);
    drawForecastChart();
    drawLossChart();
  });

  const fc = document.querySelector(".canvas-container");
  if (fc) resizeObserver.observe(fc);
}

function resizeCanvas(canvas) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
}

function drawForecastChart() {
  const canvas = dom.forecastCanvas;
  if (!canvas || !canvas.width) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  if (!state.series || state.series.length === 0) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No time-series sequence loaded", w / 2, h / 2);
    return;
  }

  const allVals = [...state.series, ...state.predictions];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const yRange = maxVal - minVal || 1.0;
  const padY = yRange * 0.14;
  const plotMinY = minVal - padY;
  const plotMaxY = maxVal + padY;

  const mLeft = 60, mRight = 35, mTop = 35, mBottom = 40;
  const plotW = w - mLeft - mRight;
  const plotH = h - mTop - mBottom;
  const totalN = allVals.length;

  function toCanvas(idx, val) {
    const cx = mLeft + (idx / Math.max(1, totalN - 1)) * plotW;
    const cy = mTop + plotH - ((val - plotMinY) / (plotMaxY - plotMinY)) * plotH;
    return [cx, cy];
  }

  // 1. Grid Lines & Coordinates
  ctx.strokeStyle = "#232838";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);

  // Y-Ticks
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  const numYTicks = 5;
  for (let i = 0; i <= numYTicks; i++) {
    const tickVal = plotMinY + (i / numYTicks) * (plotMaxY - plotMinY);
    const [, cy] = toCanvas(0, tickVal);
    ctx.beginPath();
    ctx.moveTo(mLeft, cy);
    ctx.lineTo(w - mRight, cy);
    ctx.stroke();
    ctx.fillText(tickVal.toFixed(2), mLeft - 8, cy);
  }

  // X-Ticks
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const stepStride = Math.max(1, Math.floor(totalN / 8));
  for (let i = 0; i < totalN; i += stepStride) {
    const [cx] = toCanvas(i, minVal);
    ctx.beginPath();
    ctx.moveTo(cx, mTop);
    ctx.lineTo(cx, mTop + plotH);
    ctx.stroke();
    const lbl = i >= state.series.length ? `t+${i - state.series.length + 1}` : `${i}`;
    ctx.fillText(lbl, cx, mTop + plotH + 8);
  }

  ctx.setLineDash([]);

  // Base Axes
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mLeft, mTop);
  ctx.lineTo(mLeft, mTop + plotH);
  ctx.lineTo(w - mRight, mTop + plotH);
  ctx.stroke();

  // 2. Area Fill under Historical Curve
  if (state.series.length >= 2) {
    const baseCy = mTop + plotH;
    const [firstCx] = toCanvas(0, state.series[0]);
    const [lastCx] = toCanvas(state.series.length - 1, state.series[state.series.length - 1]);

    const grad = ctx.createLinearGradient(0, mTop, 0, baseCy);
    grad.addColorStop(0, "rgba(56, 189, 248, 0.22)");
    grad.addColorStop(1, "rgba(15, 23, 42, 0.0)");

    ctx.beginPath();
    ctx.moveTo(firstCx, baseCy);
    for (let i = 0; i < state.series.length; i++) {
      const [cx, cy] = toCanvas(i, state.series[i]);
      ctx.lineTo(cx, cy);
    }
    ctx.lineTo(lastCx, baseCy);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // 3. Historical Series Curve
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < state.series.length; i++) {
    const [cx, cy] = toCanvas(i, state.series[i]);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // 4. Observation Points
  for (let i = 0; i < state.series.length; i++) {
    const [cx, cy] = toCanvas(i, state.series[i]);
    const isLast = i === state.series.length - 1;
    ctx.beginPath();
    ctx.arc(cx, cy, isLast ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fillStyle = isLast ? "#f59e0b" : "#38bdf8";
    ctx.fill();
    ctx.strokeStyle = "#0b0f19";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // 5. Forecast Trajectory & Predicted Points
  if (state.predictions && state.predictions.length > 0) {
    let [prevCx, prevCy] = toCanvas(state.series.length - 1, state.series[state.series.length - 1]);

    for (let fIdx = 0; fIdx < state.predictions.length; fIdx++) {
      const idx = state.series.length + fIdx;
      const fVal = state.predictions[fIdx];
      const [fcx, fcy] = toCanvas(idx, fVal);

      // Dashed connector
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(prevCx, prevCy);
      ctx.lineTo(fcx, fcy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Outer Glowing Ring
      ctx.beginPath();
      ctx.arc(fcx, fcy, 9, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Inner Target Dot
      ctx.beginPath();
      ctx.arc(fcx, fcy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#10b981";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Floating Callout Badge
      drawCalloutBadge(ctx, fcx, fcy - 18, `★ Next: ${fVal.toFixed(4)}`);

      prevCx = fcx;
      prevCy = fcy;
    }
  }

  // Chart Title
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 13px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("Time Series Sequence vs. RNN Forecast", mLeft, 14);
}

function drawCalloutBadge(ctx, x, y, text) {
  ctx.font = "bold 11px 'JetBrains Mono', monospace";
  const metrics = ctx.measureText(text);
  const w = metrics.width + 12;
  const h = 20;

  ctx.fillStyle = "#065f46";
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y - h / 2, w, h, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#a7f3d0";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
}

function drawLossChart() {
  const canvas = dom.lossCanvas;
  if (!canvas || !canvas.width) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  if (!state.lossHistory || state.lossHistory.length === 0) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No training loss curve data available yet", w / 2, h / 2);
    return;
  }

  const epochs = state.lossHistory.map(d => d.epoch);
  const losses = state.lossHistory.map(d => d.loss);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const maxEp = Math.max(...epochs);

  const mLeft = 70, mRight = 35, mTop = 35, mBottom = 40;
  const plotW = w - mLeft - mRight;
  const plotH = h - mTop - mBottom;

  const lRange = maxLoss - minLoss || 1.0;
  const plotMinL = Math.max(0.0, minLoss - lRange * 0.1);
  const plotMaxL = maxLoss + lRange * 0.1;

  function toCanvas(ep, lossVal) {
    const cx = mLeft + ((ep - 1) / Math.max(1, maxEp - 1)) * plotW;
    const cy = mTop + plotH - ((lossVal - plotMinL) / (plotMaxL - plotMinL)) * plotH;
    return [cx, cy];
  }

  // Grid
  ctx.strokeStyle = "#232838";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 5; i++) {
    const tickVal = plotMinL + (i / 5.0) * (plotMaxL - plotMinL);
    const [, cy] = toCanvas(1, tickVal);
    ctx.beginPath();
    ctx.moveTo(mLeft, cy);
    ctx.lineTo(w - mRight, cy);
    ctx.stroke();
    ctx.fillText(tickVal.toFixed(6), mLeft - 8, cy);
  }

  // X Epoch Ticks
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const epStride = Math.max(1, Math.floor(maxEp / 5));
  for (let ep = 0; ep <= maxEp; ep += epStride) {
    const actualEp = ep === 0 ? 1 : ep;
    const [cx] = toCanvas(actualEp, minLoss);
    ctx.beginPath();
    ctx.moveTo(cx, mTop);
    ctx.lineTo(cx, mTop + plotH);
    ctx.stroke();
    ctx.fillText(`${actualEp}`, cx, mTop + plotH + 8);
  }
  ctx.setLineDash([]);

  // Base Axes
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mLeft, mTop);
  ctx.lineTo(mLeft, mTop + plotH);
  ctx.lineTo(w - mRight, mTop + plotH);
  ctx.stroke();

  // Loss Curve
  ctx.strokeStyle = "#f43f5e";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  state.lossHistory.forEach((pt, idx) => {
    const [cx, cy] = toCanvas(pt.epoch, pt.loss);
    if (idx === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Final Target Point
  const lastPt = state.lossHistory[state.lossHistory.length - 1];
  const [endCx, endCy] = toCanvas(lastPt.epoch, lastPt.loss);
  ctx.beginPath();
  ctx.arc(endCx, endCy, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = "#10b981";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Header Title
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 13px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`Training Loss Convergence (Final MSE: ${lastPt.loss.toFixed(6)})`, mLeft, 14);
}

// Mouse Hover Handler (Strict Hover Isolation - Never Vanishes Static Plot)
function onForecastCanvasHover(e) {
  if (state.activeTab !== "forecast" || !state.series || state.series.length === 0) return;

  const canvas = dom.forecastCanvas;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const mLeft = 60, mRight = 35, mTop = 35, mBottom = 40;
  const plotW = rect.width - mLeft - mRight;

  if (mouseX < mLeft || mouseX > rect.width - mRight) {
    onForecastCanvasLeave();
    return;
  }

  const allVals = [...state.series, ...state.predictions];
  const totalN = allVals.length;
  const ratio = (mouseX - mLeft) / plotW;
  let idx = Math.round(ratio * (totalN - 1));
  idx = Math.max(0, Math.min(totalN - 1, idx));

  const val = allVals[idx];
  const isForecast = idx >= state.series.length;
  const tag = isForecast ? `Forecast t+${idx - state.series.length + 1}` : `Historical Step ${idx}`;

  dom.chartTooltip.classList.remove("hidden");
  dom.chartTooltip.style.left = `${mouseX}px`;
  dom.chartTooltip.style.top = `${mouseY}px`;
  dom.chartTooltip.innerHTML = `<strong>[${tag}]</strong> Value: <strong>${val.toFixed(4)}</strong>`;
}

function onForecastCanvasLeave() {
  dom.chartTooltip.classList.add("hidden");
}

// ==============================================================================
// Reporting & Exporting
// ==============================================================================
function copyReportToClipboard() {
  if (!state.lastResult) {
    alert("No prediction result available. Please train the model first.");
    return;
  }
  const r = state.lastResult;
  const report = `==================================================
NeuroCast RNN Time Series Prediction Report
Concept & Design by K.S.Chandra 2026
==================================================
Engine Used         : ${r.engine_used}
Epochs Completed    : ${r.epochs_completed}
Final Training Loss : ${r.final_loss}
Last Observed Value : ${r.last_val}
Predicted Value     : ${r.predicted_val}
Expected Delta      : ${r.direction} of ${Math.abs(r.diff).toFixed(4)} (${r.pct_change}%)
==================================================`;

  navigator.clipboard.writeText(report).then(() => {
    alert("Report copied to clipboard!\nConcept & Design by K.S.Chandra 2026");
  });
}

function exportForecastCsv() {
  if (!state.lastResult) {
    alert("No prediction result available to export.");
    return;
  }
  let csv = "# NeuroCast Prediction Report - Concept & Design by K.S.Chandra 2026\n";
  csv += "Step_Type,Index,Value\n";
  state.series.forEach((v, i) => {
    csv += `Historical,${i},${v}\n`;
  });
  state.predictions.forEach((v, i) => {
    csv += `Forecast,${state.series.length + i},${v}\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `neurocast_forecast_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", initApp);
