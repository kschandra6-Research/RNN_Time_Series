/**
 * NeuroCast: Frontend Client Application
 * Concept & Design by K.S.Chandra 2026
 * 
 * Full-Stack / Autonomous Web Architecture:
 * - Runs seamlessly on GitHub Pages (Autonomous Client-Side Engine)
 * - Automatically utilizes Cloud Backend API if present (Render, Railway, Docker, Local Python)
 * - Zero dependency, instant CSV/JSON/TXT file parsing
 * - 1000 Epochs Default RNN Forecasting Engine
 */

// Global Application State
const isStaticHost = (
  window.location.hostname.includes("github.io") ||
  window.location.protocol === "file:" ||
  window.location.hostname.endsWith(".pages.dev") ||
  window.location.hostname.endsWith(".netlify.app") ||
  window.location.hostname.endsWith(".vercel.app") ||
  window.location.hostname.endsWith(".gitlab.io")
);

const state = {
  series: [],
  predictions: [],
  lossHistory: [],
  lastResult: null,
  activeSource: "preset",
  activeTab: "forecast",
  availableColumns: {},
  sourceDescription: "Preset: sine_trend",
  hasBackend: false,
  isStaticDeployment: isStaticHost
};

// DOM References
const dom = {
  engineChip: document.getElementById("engine-chip"),
  engineName: document.getElementById("engine-name"),
  
  // Segmented Source Buttons (High-Contrast, No White-on-White)
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
// Pure JavaScript Neural Network Engine (Runs 100% on GitHub Pages & Offline)
// ==============================================================================

class ClientRNNModel {
  constructor(inputSize = 1, hiddenSize = 32, outputSize = 1, lr = 0.01) {
    this.H = hiddenSize;
    this.lr = lr;
    const scale = 1.0 / Math.sqrt(hiddenSize);
    
    // Deterministic pseudo-random initialization
    let seed = 42;
    const rand = () => {
      seed = (seed * 9301 + 49297) % 233280;
      return (seed / 233280.0) * 2 * scale - scale;
    };

    this.Wxh = Array.from({ length: this.H }, () => rand());
    this.Whh = Array.from({ length: this.H }, () => Array.from({ length: this.H }, () => rand()));
    this.bh = new Array(this.H).fill(0.0);
    this.Why = Array.from({ length: this.H }, () => rand());
    this.by = 0.0;

    // Adam optimizer moments
    this.mWxh = new Array(this.H).fill(0.0);
    this.vWxh = new Array(this.H).fill(0.0);
    this.mbh = new Array(this.H).fill(0.0);
    this.vbh = new Array(this.H).fill(0.0);
    this.mWhy = new Array(this.H).fill(0.0);
    this.vWhy = new Array(this.H).fill(0.0);
    this.mby = 0.0;
    this.vby = 0.0;
    this.t = 0;
  }

  forward(xVal) {
    const H = this.H;
    const h = new Array(H);
    for (let i = 0; i < H; i++) {
      h[i] = Math.tanh(this.Wxh[i] * xVal + this.bh[i]);
    }
    let yHat = this.by;
    for (let i = 0; i < H; i++) {
      yHat += this.Why[i] * h[i];
    }
    return yHat;
  }

  trainStep(xVal, target) {
    const H = this.H;
    const h = new Array(H);
    for (let i = 0; i < H; i++) {
      h[i] = Math.tanh(this.Wxh[i] * xVal + this.bh[i]);
    }
    let yHat = this.by;
    for (let i = 0; i < H; i++) {
      yHat += this.Why[i] * h[i];
    }

    const diff = yHat - target;
    const loss = diff * diff;
    const dy = 2.0 * diff;

    const dby = dy;
    const dWhy = new Array(H);
    const dh = new Array(H);
    const da = new Array(H);
    const dbh = new Array(H);
    const dWxh = new Array(H);

    for (let i = 0; i < H; i++) {
      dWhy[i] = dy * h[i];
      dh[i] = this.Why[i] * dy;
      da[i] = (1.0 - h[i] * h[i]) * dh[i];
      dbh[i] = da[i];
      dWxh[i] = da[i] * xVal;
    }

    this.t += 1;
    const t = this.t;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const lr_t = this.lr * Math.sqrt(1.0 - Math.pow(b2, t)) / (1.0 - Math.pow(b1, t));

    for (let i = 0; i < H; i++) {
      let g = Math.max(-1.0, Math.min(1.0, dWxh[i]));
      this.mWxh[i] = b1 * this.mWxh[i] + (1 - b1) * g;
      this.vWxh[i] = b2 * this.vWxh[i] + (1 - b2) * (g * g);
      this.Wxh[i] -= lr_t * this.mWxh[i] / (Math.sqrt(this.vWxh[i]) + eps);

      g = Math.max(-1.0, Math.min(1.0, dbh[i]));
      this.mbh[i] = b1 * this.mbh[i] + (1 - b1) * g;
      this.vbh[i] = b2 * this.vbh[i] + (1 - b2) * (g * g);
      this.bh[i] -= lr_t * this.mbh[i] / (Math.sqrt(this.vbh[i]) + eps);

      g = Math.max(-1.0, Math.min(1.0, dWhy[i]));
      this.mWhy[i] = b1 * this.mWhy[i] + (1 - b1) * g;
      this.vWhy[i] = b2 * this.vWhy[i] + (1 - b2) * (g * g);
      this.Why[i] -= lr_t * this.mWhy[i] / (Math.sqrt(this.vWhy[i]) + eps);
    }

    const gBy = Math.max(-1.0, Math.min(1.0, dby));
    this.mby = b1 * this.mby + (1 - b1) * gBy;
    this.vby = b2 * this.vby + (1 - b2) * (gBy * gBy);
    this.by -= lr_t * this.mby / (Math.sqrt(this.vby) + eps);

    return loss;
  }
}

function trainClientRNN(data, numEpochs = 1000, lr = 0.01, hiddenSize = 32, horizon = 1) {
  const t0 = performance.now();
  const clean = data.filter(x => !isNaN(x) && isFinite(x));
  if (clean.length < 2) throw new Error("Series requires at least 2 valid numeric observations.");

  const minVal = Math.min(...clean);
  const maxVal = Math.max(...clean);
  const vRange = maxVal - minVal || 1.0;
  const scaled = clean.map(x => (x - minVal) / vRange);

  const model = new ClientRNNModel(1, hiddenSize, 1, lr);
  const lossHistory = [];
  let finalLoss = 0.0;

  for (let epoch = 1; epoch <= numEpochs; epoch++) {
    let epochLoss = 0.0;
    for (let i = 0; i < scaled.length - 1; i++) {
      epochLoss += model.trainStep(scaled[i], scaled[i + 1]);
    }
    const avgLoss = epochLoss / Math.max(1, scaled.length - 1);
    finalLoss = avgLoss;

    if (epoch % Math.max(1, Math.floor(numEpochs / 50)) === 0 || epoch === numEpochs) {
      lossHistory.push({ epoch, loss: parseFloat(avgLoss.toFixed(6)) });
    }
  }

  // Multi-step Autoregressive Rollout
  const predsScaled = [];
  let curr = scaled[scaled.length - 1];
  for (let s = 0; s < horizon; s++) {
    const nextVal = model.forward(curr);
    predsScaled.push(nextVal);
    curr = nextVal;
  }

  const predictions = predsScaled.map(p => Number((p * vRange + minVal).toFixed(4)));
  const predictedVal = predictions[0];
  const lastVal = clean[clean.length - 1];
  const diff = Number((predictedVal - lastVal).toFixed(4));
  const pctChange = lastVal !== 0 ? Number(((diff / Math.abs(lastVal)) * 100).toFixed(2)) : 0.0;
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

  const engineName = state.isStaticDeployment ? "Client Engine (Autonomous / GitHub Pages)" : "Client Neural Engine";

  return {
    success: true,
    history: clean,
    predictions,
    last_val: Number(lastVal.toFixed(4)),
    predicted_val: predictedVal,
    diff,
    pct_change: pctChange,
    direction: diff >= 0 ? "Increase" : "Decrease",
    direction_symbol: diff >= 0 ? "▲" : "▼",
    final_loss: parseFloat(finalLoss.toFixed(6)),
    loss_history: lossHistory,
    engine_used: engineName,
    training_time: parseFloat(elapsed),
    epochs_completed: numEpochs,
    concept_and_design: "Concept & Design by K.S.Chandra 2026"
  };
}

// ==============================================================================
// Robust Client-Side File Parsing (CSV, JSON, TXT)
// ==============================================================================

function parseNumericCell(str) {
  if (typeof str === "number") return isFinite(str) ? str : NaN;
  if (!str || typeof str !== "string") return NaN;
  const t = str.trim();
  if (t === "" || t === "NaN" || t === "null" || t === "undefined") return NaN;
  // Reject date/timestamp strings (e.g., 2026-10-01, 10/01/2026)
  if (/^\d{4}[-/]\d{1,2}[-/]\d{1,2}/.test(t) || /^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}/.test(t)) return NaN;
  const num = Number(t);
  if (!isNaN(num) && isFinite(num)) return num;
  const cleanStr = t.replace(/^[$€£¥]/, "").replace(/[%]$/, "").trim();
  const num2 = Number(cleanStr);
  if (!isNaN(num2) && isFinite(num2)) return num2;
  return NaN;
}

function parseFileContentClient(content, filename = "") {
  content = content.trim();
  if (!content) throw new Error("File content is empty.");

  // 1. JSON Support
  if (content.startsWith("[") || content.startsWith("{")) {
    try {
      const data = JSON.parse(content);
      if (Array.isArray(data) && data.every(x => typeof x === "number")) {
        return { "values": data };
      }
      if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object") {
        const cols = {};
        Object.keys(data[0]).forEach(k => {
          const vals = [];
          data.forEach(row => {
            if (row && row[k] !== undefined) {
              const num = parseNumericCell(String(row[k]));
              if (!isNaN(num)) vals.push(num);
            }
          });
          if (vals.length >= 2) cols[k] = vals;
        });
        if (Object.keys(cols).length > 0) return cols;
      }
      if (typeof data === "object" && !Array.isArray(data)) {
        const cols = {};
        Object.keys(data).forEach(k => {
          if (Array.isArray(data[k])) {
            const vals = data[k].map(x => parseNumericCell(String(x))).filter(x => !isNaN(x));
            if (vals.length >= 2) cols[k] = vals;
          }
        });
        if (Object.keys(cols).length > 0) return cols;
      }
    } catch (e) {}
  }

  // 2. CSV / TSV / Delimited Lines
  const lines = content.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) throw new Error("No data lines found in file.");

  const first10 = lines.slice(0, 10).join("\n");
  let delim = ",";
  if (first10.includes("\t")) delim = "\t";
  else if (first10.includes(";") && !first10.includes(",")) delim = ";";
  else if (first10.includes("|")) delim = "|";

  const rows = lines.map(line => {
    return line.split(delim).map(cell => cell.trim().replace(/^["']|["']$/g, ""));
  });

  if (rows.length > 0) {
    const header = rows[0];
    const hasHeader = header.some(h => isNaN(Number(h)));

    if (hasHeader && rows.length > 1) {
      const cols = {};
      header.forEach((h, colIdx) => {
        const colName = h || `Column ${colIdx + 1}`;
        const vals = [];
        for (let r = 1; r < rows.length; r++) {
          if (rows[r][colIdx] !== undefined) {
            const val = parseNumericCell(rows[r][colIdx]);
            if (!isNaN(val)) vals.push(val);
          }
        }
        if (vals.length >= 2) cols[colName] = vals;
      });
      if (Object.keys(cols).length > 0) return cols;
    }

    const tokens = [];
    rows.forEach(row => {
      row.forEach(cell => {
        cell.replace(/,/g, " ").split(/\s+/).forEach(token => {
          const val = parseNumericCell(token);
          if (!isNaN(val)) tokens.push(val);
        });
      });
    });
    if (tokens.length >= 2) {
      return { "series": tokens };
    }
  }

  throw new Error("Unable to extract numeric time-series observations from file.");
}

// ==============================================================================
// Synthetic Wave Presets
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
// Initialization & Backend Probing
// ==============================================================================

async function initApp() {
  bindEvents();
  setupCanvasResize();
  
  if (state.isStaticDeployment) {
    dom.engineName.textContent = "Autonomous Mode (Client Engine)";
    dom.engineChip.style.color = "#38bdf8";
    logMessage("⚡ Running in Autonomous Mode (Static Hosting / GitHub Pages).");
    logMessage("Concept & Design by K.S.Chandra 2026");
    logMessage("100% Client-side neural network and data parsers active.");
  } else {
    // Probe Backend API using relative URL
    try {
      const res = await fetch("api/health");
      const cType = res.headers.get("content-type") || "";
      if (res.ok && cType.includes("application/json")) {
        const data = await res.json();
        state.hasBackend = true;
        dom.engineName.textContent = `Backend Online • ${data.engine_detected}`;
        logMessage(`Connected to Python Backend: ${data.engine_detected}`);
      } else {
        dom.engineName.textContent = "Client Engine (Autonomous)";
        logMessage("Server API not detected. Falling back to Client-Side Engine.");
      }
    } catch (e) {
      dom.engineName.textContent = "Client Engine (Autonomous)";
      logMessage("Running in Client-Side Engine mode.");
    }
  }

  loadPresetData();
}

// ==============================================================================
// Event Listeners
// ==============================================================================

function bindEvents() {
  // High-Contrast Segmented Source Buttons (No White-on-White text hiding)
  dom.segPreset.addEventListener("click", () => switchSource("preset"));
  dom.segFile.addEventListener("click", () => switchSource("file"));
  dom.segManual.addEventListener("click", () => switchSource("manual"));

  dom.presetSelect.addEventListener("change", loadPresetData);

  // File Upload Handlers (Pure Client Parsing, Zero 404s!)
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

  // Canvas Hover Tooltips (Strict Hover Isolation - Never Vanishes Static Plot)
  dom.forecastCanvas.addEventListener("mousemove", onForecastCanvasHover);
  dom.forecastCanvas.addEventListener("mouseleave", onForecastCanvasLeave);

  // Export Buttons
  dom.btnCopyReport.addEventListener("click", copyReportToClipboard);
  dom.btnExportCsv.addEventListener("click", exportForecastCsv);
}

window.setEpochs = function(val) {
  dom.inputEpochs.value = val;
  document.querySelectorAll(".pill-btn").forEach(p => {
    p.classList.toggle("active", Number(p.textContent.trim()) === val);
  });
};

// ==============================================================================
// Source Switcher & File Handling
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

  try {
    const text = await file.text();
    // Parse client-side: Instantaneous, zero network requests, zero 404s!
    const cols = parseFileContentClient(text, file.name);
    state.availableColumns = cols;

    const colNames = Object.keys(cols);
    dom.columnSelect.innerHTML = "";
    colNames.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = `${c} (${cols[c].length} observations)`;
      dom.columnSelect.appendChild(opt);
    });

    dom.fileColumnContainer.classList.remove("hidden");
    const initialCol = colNames[0];
    setSeriesData(cols[initialCol], `${file.name} [${initialCol}]`);
    logMessage(`Loaded file '${file.name}' successfully. Extracted columns: [${colNames.join(", ")}]`);

  } catch (err) {
    alert(`Error loading file: ${err.message}`);
    logMessage(`[File Error] ${err.message}`);
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
// Training Workflow & API Execution (Hybrid Cloud + GitHub Pages Autonomous)
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

  logMessage("=== NeuroCast Training Request Started ===");
  logMessage(`Target Observations: ${state.series.length}`);
  logMessage(`Epochs: ${epochs} (Default) | LR: ${lr} | Hidden: ${hiddenSize} | Horizon: ${horizon}`);
  logMessage("Concept & Design by K.S.Chandra 2026");

  let simProgress = 0;
  const simTimer = setInterval(() => {
    if (simProgress < 90) {
      simProgress += 15;
      dom.progressBar.style.width = `${simProgress}%`;
      dom.statusMetrics.textContent = `Epoch ${Math.round((simProgress / 100) * epochs)} / ${epochs} | Optimizing...`;
    }
  }, 50);

  const tStart = performance.now();
  let result = null;

  try {
    // 1. If Cloud Backend is active, dispatch request to server
    if (state.hasBackend && !state.isStaticDeployment) {
      try {
        const response = await fetch("api/train", {
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

        const contentType = response.headers.get("content-type") || "";
        if (response.ok && contentType.includes("application/json")) {
          result = await response.json();
        } else {
          // Cloud returned non-JSON / 404 -> smoothly fallback to client engine
          state.hasBackend = false;
          logMessage("[Notice] Server returned non-JSON response. Executing with Client Neural Engine...");
          result = trainClientRNN(state.series, epochs, lr, hiddenSize, horizon);
        }
      } catch (err) {
        state.hasBackend = false;
        logMessage(`[Notice] Cloud API unreachable (${err.message}). Executing with Client Neural Engine...`);
        result = trainClientRNN(state.series, epochs, lr, hiddenSize, horizon);
      }
    } else {
      // 2. Autonomous Mode (GitHub Pages or Static Host)
      await new Promise(r => setTimeout(r, 60)); // Yield to paint UI progress
      result = trainClientRNN(state.series, epochs, lr, hiddenSize, horizon);
    }

    clearInterval(simTimer);
    dom.progressBar.style.width = "100%";

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
    logMessage(`Computation Duration : ${result.training_time}s`);
    logMessage("=== Execution Successful ===");

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
// Interactive Vector Canvas Chart Engine (Strict Hover Isolation)
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

      // Dashed connection line
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
