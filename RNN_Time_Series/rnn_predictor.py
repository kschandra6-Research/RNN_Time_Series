"""
NeuroCast: Interactive Time Series RNN Predictor
=================================================
Concept & Design by K.S.Chandra 2026

A cross-platform, zero-dependency, professional Time Series RNN forecasting application.

Features:
- Professional Tkinter GUI with custom dark modern theme and dynamic canvas charting.
- High-contrast segmented button controls for Data Source selection (No white-on-white hiding).
- Safe vector Canvas plotting with isolated hover tooltips (Historical & Forecast never vanish).
- Triple Engine Architecture:
    1. PyTorch Engine (Accelerated GPU/CPU, if installed)
    2. NumPy Engine (Vectorized matrix math, if installed)
    3. Pure Python Engine (Built-in standard library, ZERO external dependencies)
- Automatic Data Parsing for CSV, TSV, JSON, TXT, and Excel (with column auto-detection).
- Multithreaded Training (non-blocking UI with real-time epoch & loss streaming).
- Default Epochs: 1000 (user-configurable).
- Seamless Cross-Platform Execution (Windows, macOS, Linux) with automatic Headless/CLI fallback.
"""

import os
import sys
import math
import csv
import json
import random
import time
import threading
import traceback
import platform
import argparse
from typing import List, Tuple, Optional, Dict, Any, Callable

# ==============================================================================
# Platform & DPI Awareness
# ==============================================================================
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

if IS_WINDOWS:
    try:
        import ctypes
        # Enable Per-Monitor High-DPI Awareness on Windows
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ==============================================================================
# Dependency Probing (Zero-Hard-Dependency Policy)
# ==============================================================================
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TKINTER_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


# ==============================================================================
# Neural Network Engines
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. PyTorch RNN Engine (Optional Acceleration)
# ------------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class TorchRNNModel(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1):
            super().__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.rnn(x, h0)
            return self.fc(out[:, -1, :])


# ------------------------------------------------------------------------------
# 2. NumPy RNN Engine (Accelerated Matrix Math with Adam Optimizer)
# ------------------------------------------------------------------------------
class NumpyRNNModel:
    def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1, lr: float = 0.01):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is not installed.")
        np.random.seed(42)
        scale = 1.0 / np.sqrt(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

        self.params = [self.Wxh, self.Whh, self.bh, self.Why, self.by]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.hs = {}
        self.xs = {}

    def forward(self, inputs: List[float]) -> float:
        h = np.zeros((self.hidden_size, 1))
        self.hs = {-1: h}
        self.xs = {}
        for t, x_val in enumerate(inputs):
            x = np.array([[x_val]], dtype=float)
            self.xs[t] = x
            h = np.tanh(self.Wxh @ x + self.Whh @ self.hs[t - 1] + self.bh)
            self.hs[t] = h
        y_hat = self.Why @ h + self.by
        return float(y_hat[0, 0])

    def train_step(self, inputs: List[float], target: float) -> float:
        y_hat = self.forward(inputs)
        loss = (y_hat - target) ** 2
        dy = 2.0 * (y_hat - target)

        last_idx = len(inputs) - 1
        dWhy = np.array([[dy]]) @ self.hs[last_idx].T
        dby = np.array([[dy]])
        dh = self.Why.T @ np.array([[dy]])
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(len(inputs))):
            da = (1.0 - self.hs[t] ** 2) * dh
            dbh += da
            dWxh += da @ self.xs[t].T
            dWhh += da @ self.hs[t - 1].T
            dh = self.Whh.T @ da

        grads = [dWxh, dWhh, dbh, dWhy, dby]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.params)):
            np.clip(grads[i], -1.0, 1.0, out=grads[i])
            self.m[i] = b1 * self.m[i] + (1 - b1) * grads[i]
            self.v[i] = b2 * self.v[i] + (1 - b2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return float(loss)


# ------------------------------------------------------------------------------
# 3. Pure Python RNN Engine (Zero External Dependencies, Standard Library Only)
# ------------------------------------------------------------------------------
class PurePythonRNNModel:
    """
    High-performance Pure Python 1-layer RNN with Adam Optimizer.
    Uses only built-in math and random modules. Runs out-of-the-box on ANY Python setup.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 24, output_size: int = 1, lr: float = 0.01):
        random.seed(42)
        scale = 1.0 / math.sqrt(hidden_size)
        self.H = hidden_size
        self.lr = lr

        # Weights
        self.Wxh = [random.uniform(-scale, scale) for _ in range(self.H)]
        self.Whh = [[random.uniform(-scale, scale) for _ in range(self.H)] for _ in range(self.H)]
        self.bh = [0.0] * self.H
        self.Why = [random.uniform(-scale, scale) for _ in range(self.H)]
        self.by = 0.0

        # Adam moments
        self.mWxh = [0.0] * self.H
        self.vWxh = [0.0] * self.H
        self.mbh = [0.0] * self.H
        self.vbh = [0.0] * self.H
        self.mWhy = [0.0] * self.H
        self.vWhy = [0.0] * self.H
        self.mby = 0.0
        self.vby = 0.0
        self.t = 0

    def forward(self, x_val: float) -> float:
        h = [0.0] * self.H
        for i in range(self.H):
            h[i] = math.tanh(self.Wxh[i] * x_val + self.bh[i])
        y_hat = self.by + sum(self.Why[i] * h[i] for i in range(self.H))
        return y_hat

    def train_step(self, x_val: float, target: float) -> float:
        H = self.H
        h = [0.0] * H
        for i in range(H):
            h[i] = math.tanh(self.Wxh[i] * x_val + self.bh[i])
        y_hat = self.by + sum(self.Why[i] * h[i] for i in range(H))

        diff = y_hat - target
        loss = diff * diff
        dy = 2.0 * diff

        dby = dy
        dWhy = [dy * h[i] for i in range(H)]
        dh = [self.Why[i] * dy for i in range(H)]
        da = [(1.0 - h[i] * h[i]) * dh[i] for i in range(H)]
        dbh = da
        dWxh = [da[i] * x_val for i in range(H)]

        self.t += 1
        t = self.t
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr_t = self.lr * math.sqrt(1.0 - (b2 ** t)) / (1.0 - (b1 ** t))

        for i in range(H):
            g = max(-1.0, min(1.0, dWxh[i]))
            self.mWxh[i] = b1 * self.mWxh[i] + (1 - b1) * g
            self.vWxh[i] = b2 * self.vWxh[i] + (1 - b2) * (g * g)
            self.Wxh[i] -= lr_t * self.mWxh[i] / (math.sqrt(self.vWxh[i]) + eps)

            g = max(-1.0, min(1.0, dbh[i]))
            self.mbh[i] = b1 * self.mbh[i] + (1 - b1) * g
            self.vbh[i] = b2 * self.vbh[i] + (1 - b2) * (g * g)
            self.bh[i] -= lr_t * self.mbh[i] / (math.sqrt(self.vbh[i]) + eps)

            g = max(-1.0, min(1.0, dWhy[i]))
            self.mWhy[i] = b1 * self.mWhy[i] + (1 - b1) * g
            self.vWhy[i] = b2 * self.vWhy[i] + (1 - b2) * (g * g)
            self.Why[i] -= lr_t * self.mWhy[i] / (math.sqrt(self.vWhy[i]) + eps)

        g = max(-1.0, min(1.0, dby))
        self.mby = b1 * self.mby + (1 - b1) * g
        self.vby = b2 * self.vby + (1 - b2) * (g * g)
        self.by -= lr_t * self.mby / (math.sqrt(self.vby) + eps)

        return loss


# ==============================================================================
# Data Scaling & Normalization
# ==============================================================================

def normalize_series(series: List[float]) -> Tuple[List[float], float, float]:
    clean = [float(x) for x in series if not (math.isnan(x) or math.isinf(x))]
    if len(clean) < 2:
        raise ValueError("Series requires at least 2 valid numeric points.")
    min_val = min(clean)
    max_val = max(clean)
    val_range = max_val - min_val if max_val != min_val else 1.0
    scaled = [(x - min_val) / val_range for x in clean]
    return scaled, min_val, val_range


def denormalize_value(scaled_val: float, min_val: float, val_range: float) -> float:
    return scaled_val * val_range + min_val


# ==============================================================================
# Model Training Routine & Result Struct
# ==============================================================================

class PredictionResult:
    def __init__(self,
                 history: List[float],
                 predictions: List[float],
                 last_val: float,
                 predicted_val: float,
                 diff: float,
                 pct_change: float,
                 final_loss: float,
                 loss_history: List[Tuple[int, float]],
                 engine_used: str,
                 training_time: float,
                 epochs_completed: int):
        self.history = history
        self.predictions = predictions
        self.last_val = last_val
        self.predicted_val = predicted_val
        self.diff = diff
        self.pct_change = pct_change
        self.final_loss = final_loss
        self.loss_history = loss_history
        self.engine_used = engine_used
        self.training_time = training_time
        self.epochs_completed = epochs_completed

    @property
    def direction(self) -> str:
        return "Increase" if self.diff >= 0 else "Decrease"

    @property
    def direction_symbol(self) -> str:
        return "▲" if self.diff >= 0 else "▼"


def train_rnn_engine(data: List[float],
                     num_epochs: int = 1000,
                     learning_rate: float = 0.01,
                     hidden_size: int = 32,
                     forecast_steps: int = 1,
                     engine_preference: str = "auto",
                     progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
                     cancel_event: Optional[threading.Event] = None) -> PredictionResult:
    """
    Train RNN model for specified epochs (default 1000) using the best available engine.
    Supports real-time progress callbacks and cancellation.
    """
    t_start = time.time()
    scaled_data, min_val, val_range = normalize_series(data)
    n_points = len(scaled_data)

    # Determine Engine
    selected_engine = "pure_python"
    pref = engine_preference.lower()
    if pref == "pytorch" and TORCH_AVAILABLE:
        selected_engine = "pytorch"
    elif pref == "numpy" and NUMPY_AVAILABLE:
        selected_engine = "numpy"
    elif pref == "pure_python":
        selected_engine = "pure_python"
    else:  # auto
        if NUMPY_AVAILABLE:
            selected_engine = "numpy"
        elif TORCH_AVAILABLE:
            selected_engine = "pytorch"
        else:
            selected_engine = "pure_python"

    loss_history: List[Tuple[int, float]] = []
    final_loss = 0.0
    completed_epochs = 0

    # --------------------------------------------------------------------------
    # Execute PyTorch Engine
    # --------------------------------------------------------------------------
    if selected_engine == "pytorch":
        engine_display = "PyTorch"
        tensor_data = torch.FloatTensor(scaled_data).view(-1)
        model = TorchRNNModel(input_size=1, hidden_size=hidden_size, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            if cancel_event and cancel_event.is_set():
                break
            epoch_loss = 0.0
            for i in range(n_points - 1):
                seq = tensor_data[i : i + 1].view(-1, 1, 1)
                tgt = tensor_data[i + 1].view(-1)
                out = model(seq)
                loss = criterion(out.view(-1), tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, n_points - 1)
            final_loss = avg_loss
            completed_epochs = epoch + 1

            if completed_epochs % max(1, num_epochs // 50) == 0 or completed_epochs == num_epochs:
                loss_history.append((completed_epochs, avg_loss))
                if progress_callback:
                    progress_callback(completed_epochs, num_epochs, avg_loss, time.time() - t_start)

        preds_scaled = []
        curr_in = tensor_data[-1:].view(-1, 1, 1)
        model.eval()
        with torch.no_grad():
            for _ in range(forecast_steps):
                p_out = model(curr_in).view(-1).item()
                preds_scaled.append(p_out)
                curr_in = torch.FloatTensor([p_out]).view(1, 1, 1)

    # --------------------------------------------------------------------------
    # Execute NumPy Engine
    # --------------------------------------------------------------------------
    elif selected_engine == "numpy":
        engine_display = "NumPy (Accelerated)"
        model = NumpyRNNModel(input_size=1, hidden_size=hidden_size, output_size=1, lr=learning_rate)

        for epoch in range(num_epochs):
            if cancel_event and cancel_event.is_set():
                break
            epoch_loss = 0.0
            for i in range(n_points - 1):
                loss = model.train_step([scaled_data[i]], scaled_data[i + 1])
                epoch_loss += loss

            avg_loss = epoch_loss / max(1, n_points - 1)
            final_loss = avg_loss
            completed_epochs = epoch + 1

            if completed_epochs % max(1, num_epochs // 50) == 0 or completed_epochs == num_epochs:
                loss_history.append((completed_epochs, avg_loss))
                if progress_callback:
                    progress_callback(completed_epochs, num_epochs, avg_loss, time.time() - t_start)

        preds_scaled = []
        curr_val = scaled_data[-1]
        for _ in range(forecast_steps):
            next_scaled = model.forward([curr_val])
            preds_scaled.append(next_scaled)
            curr_val = next_scaled

    # --------------------------------------------------------------------------
    # Execute Pure Python Engine (Zero Dependencies)
    # --------------------------------------------------------------------------
    else:
        engine_display = "Pure Python (Built-in)"
        model = PurePythonRNNModel(input_size=1, hidden_size=min(hidden_size, 32), output_size=1, lr=learning_rate)

        for epoch in range(num_epochs):
            if cancel_event and cancel_event.is_set():
                break
            epoch_loss = 0.0
            for i in range(n_points - 1):
                loss = model.train_step(scaled_data[i], scaled_data[i + 1])
                epoch_loss += loss

            avg_loss = epoch_loss / max(1, n_points - 1)
            final_loss = avg_loss
            completed_epochs = epoch + 1

            if completed_epochs % max(1, num_epochs // 50) == 0 or completed_epochs == num_epochs:
                loss_history.append((completed_epochs, avg_loss))
                if progress_callback:
                    progress_callback(completed_epochs, num_epochs, avg_loss, time.time() - t_start)

        preds_scaled = []
        curr_val = scaled_data[-1]
        for _ in range(forecast_steps):
            next_scaled = model.forward(curr_val)
            preds_scaled.append(next_scaled)
            curr_val = next_scaled

    # Denormalize predictions
    predictions = [denormalize_value(p, min_val, val_range) for p in preds_scaled]
    predicted_val = predictions[0]
    last_val = data[-1]
    diff = predicted_val - last_val
    pct_change = (diff / abs(last_val) * 100.0) if last_val != 0 else 0.0
    elapsed = time.time() - t_start

    return PredictionResult(
        history=data,
        predictions=predictions,
        last_val=last_val,
        predicted_val=predicted_val,
        diff=diff,
        pct_change=pct_change,
        final_loss=final_loss,
        loss_history=loss_history,
        engine_used=engine_display,
        training_time=elapsed,
        epochs_completed=completed_epochs
    )


# ==============================================================================
# Robust File Loading Engine (Zero-Dependency + Optional Pandas Enhancement)
# ==============================================================================

def inspect_file_columns(filepath: str) -> Dict[str, Any]:
    """
    Detect available columns and sample numeric series from CSV, JSON, TXT, or Excel.
    Works natively with standard library.
    """
    ext = os.path.splitext(filepath)[1].lower()
    columns_info = {}

    if ext in [".xlsx", ".xls"]:
        if not PANDAS_AVAILABLE:
            raise ValueError("Reading Excel files (.xlsx/.xls) requires 'pandas'. Use .csv, .json, or .txt for zero-dependency parsing.")
        try:
            df = pd.read_excel(filepath)
            for col in df.columns:
                num_series = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
                columns_info[str(col)] = num_series
            return columns_info
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

    if ext == ".json":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                columns_info["values"] = [float(x) for x in data]
                return columns_info
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                keys = list(data[0].keys())
                for k in keys:
                    vals = []
                    for row in data:
                        if isinstance(row, dict) and k in row:
                            try:
                                vals.append(float(row[k]))
                            except (ValueError, TypeError):
                                pass
                    if vals:
                        columns_info[k] = vals
                return columns_info
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        vals = []
                        for item in v:
                            try:
                                vals.append(float(item))
                            except (ValueError, TypeError):
                                pass
                        if vals:
                            columns_info[k] = vals
                return columns_info
        except Exception as e:
            raise ValueError(f"Error parsing JSON: {e}")

    # Default CSV/TSV/TXT parser (Standard library)
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc, errors="replace") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue

            first_chunk = "\n".join(lines[:10])
            delimiter = None
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(first_chunk, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except Exception:
                if "\t" in first_chunk:
                    delimiter = "\t"
                elif ";" in first_chunk:
                    delimiter = ";"
                elif "," in first_chunk:
                    delimiter = ","

            if delimiter:
                reader = list(csv.reader(lines, delimiter=delimiter))
            else:
                reader = [[item for item in line.split() if item] for line in lines]

            if not reader:
                continue

            header = reader[0]
            has_header = False
            for token in header:
                try:
                    float(token)
                except ValueError:
                    has_header = True
                    break

            if has_header and len(reader) > 1:
                cols = {h.strip(): [] for h in header if h.strip()}
                for row in reader[1:]:
                    for h, val in zip(header, row):
                        h = h.strip()
                        if h in cols:
                            try:
                                cols[h].append(float(val.strip()))
                            except ValueError:
                                pass
                for h, vals in cols.items():
                    if vals:
                        columns_info[h] = vals
                if columns_info:
                    return columns_info

            tokens = []
            for row in reader:
                for item in row:
                    for sub in item.replace(",", " ").split():
                        try:
                            tokens.append(float(sub))
                        except ValueError:
                            pass
            if len(tokens) >= 2:
                columns_info["series"] = tokens
                return columns_info

        except Exception:
            continue

    if not columns_info:
        raise ValueError("No numeric data columns could be extracted from this file.")
    return columns_info


def find_workspace_data_files() -> List[str]:
    """Scan script directory and current working directory for compatible time series files."""
    candidates = []
    seen = set()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [script_dir]
    cwd = os.getcwd()
    if cwd != script_dir and not any(part.lower() in cwd.lower() for part in ["program files", "programs", "appdata"]):
        dirs.append(cwd)

    extensions = {".csv", ".json", ".txt", ".tsv", ".xlsx", ".xls"}

    for d in dirs:
        if os.path.isdir(d):
            try:
                for entry in sorted(os.listdir(d)):
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in extensions and not entry.startswith("."):
                        lower_name = entry.lower()
                        if any(skip in lower_name for skip in ["requirement", "license", "setup", "readme"]):
                            continue
                        full = os.path.join(d, entry)
                        if full not in seen and os.path.isfile(full):
                            try:
                                cols = inspect_file_columns(full)
                                if cols:
                                    seen.add(full)
                                    candidates.append(full)
                            except Exception:
                                pass
            except Exception:
                pass
    return candidates


def generate_synthetic_series(preset_type: str, n_steps: int = 50) -> List[float]:
    """Generate clean synthetic time series patterns for instant demonstration."""
    if preset_type == "sine_trend":
        return [round(math.sin(i * 0.25) * 4.0 + (i * 0.4) + 12.0, 2) for i in range(n_steps)]
    elif preset_type == "stock_walk":
        random.seed(101)
        price = 100.0
        series = [price]
        for _ in range(n_steps - 1):
            shock = random.gauss(0.15, 1.2)
            price = max(10.0, price + shock)
            series.append(round(price, 2))
        return series
    elif preset_type == "seasonal_sales":
        return [round(15.0 + 3.0 * math.sin(i * 0.6) + 6.0 * math.cos(i * 0.15) + i * 0.2, 2) for i in range(n_steps)]
    elif preset_type == "damped_oscillation":
        return [round(20.0 + 15.0 * math.exp(-i * 0.05) * math.cos(i * 0.4), 2) for i in range(n_steps)]
    else:
        return [round(10.0 + i * 1.5 + (random.uniform(-0.5, 0.5)), 2) for i in range(n_steps)]


# ==============================================================================
# Custom High-Contrast Segmented Button Bar (Fixes White Color Issue)
# ==============================================================================

if TKINTER_AVAILABLE:
    class SegmentedButtonBar(tk.Frame):
        """
        Modern high-contrast segmented toggle button bar.
        Eliminates default OS radiobutton artifacts that turn white and hide text options.
        """
        def __init__(self, parent, options: List[Tuple[str, str]], on_change: Optional[Callable[[str], None]] = None, **kwargs):
            super().__init__(parent, bg="#0f172a", highlightthickness=1, highlightbackground="#334155", **kwargs)
            self.options = options
            self.on_change = on_change
            self.selected_value = options[0][0]
            self.buttons: Dict[str, tk.Button] = {}

            for val, label in options:
                btn = tk.Button(
                    self, text=label, font=("Segoe UI", 9, "bold"),
                    relief="flat", bd=0, padx=12, pady=6, cursor="hand2",
                    command=lambda v=val: self.select(v)
                )
                btn.pack(side="left", padx=2, pady=2, fill="x", expand=True)
                btn.bind("<Enter>", lambda e, v=val: self._on_hover(v, True))
                btn.bind("<Leave>", lambda e, v=val: self._on_hover(v, False))
                self.buttons[val] = btn

            self._update_styles()

        def select(self, val: str):
            if val != self.selected_value:
                self.selected_value = val
                self._update_styles()
                if self.on_change:
                    self.on_change(val)

        def _update_styles(self):
            for val, btn in self.buttons.items():
                if val == self.selected_value:
                    # Active selected state: Vibrant Blue with Crisp White Text
                    btn.configure(
                        bg="#2563eb", fg="#ffffff",
                        activebackground="#1d4ed8", activeforeground="#ffffff"
                    )
                else:
                    # Inactive unselected state: Dark Slate with High-Contrast Light Slate Text
                    btn.configure(
                        bg="#1e293b", fg="#cbd5e1",
                        activebackground="#334155", activeforeground="#ffffff"
                    )

        def _on_hover(self, val: str, is_enter: bool):
            btn = self.buttons.get(val)
            if not btn:
                return
            if val != self.selected_value:
                if is_enter:
                    btn.configure(bg="#334155", fg="#ffffff")
                else:
                    btn.configure(bg="#1e293b", fg="#cbd5e1")


# ==============================================================================
# Zero-Dependency High-Quality Canvas Plot Engine
# ==============================================================================

if TKINTER_AVAILABLE:
    class CanvasPlot(tk.Canvas):
        """
        Interactive 2D charting engine built entirely on standard Tkinter Canvas.
        Provides smooth line rendering, area fills, glowing prediction markers, gridlines,
        and mouse hover inspection with strict tag isolation (historical and forecast items
        never vanish on mouse hover).
        """
        def __init__(self, parent, bg="#111827", **kwargs):
            super().__init__(parent, bg=bg, highlightthickness=0, **kwargs)
            self.bg_color = bg
            self.history: List[float] = []
            self.forecast: List[float] = []
            self.loss_data: List[Tuple[int, float]] = []
            self.mode = "time_series"

            self.bind("<Configure>", lambda e: self.redraw())
            self.bind("<Motion>", self._on_mouse_move)
            self.bind("<Leave>", self._on_mouse_leave)

        def set_time_series(self, history: List[float], forecast: Optional[List[float]] = None):
            self.mode = "time_series"
            self.history = history
            self.forecast = forecast or []
            self.redraw()

        def set_loss_curve(self, loss_data: List[Tuple[int, float]]):
            self.mode = "loss_curve"
            self.loss_data = loss_data
            self.redraw()

        def redraw(self):
            self.delete("all")
            w = self.winfo_width()
            h = self.winfo_height()
            if w < 100:
                try:
                    w = int(self.cget("width"))
                except Exception:
                    pass
            if h < 80:
                try:
                    h = int(self.cget("height"))
                except Exception:
                    pass
            if w < 100 or h < 80:
                return

            if self.mode == "time_series":
                self._draw_time_series(w, h)
            else:
                self._draw_loss_curve(w, h)

        def _draw_time_series(self, w: int, h: int):
            if not self.history:
                self.create_text(w // 2, h // 2, text="No time series data loaded", fill="#64748b", font=("Segoe UI", 12), tags="static_plot")
                return

            all_vals = self.history + self.forecast
            min_y, max_y = min(all_vals), max(all_vals)
            y_range = max_y - min_y if max_y != min_y else 1.0
            pad_y = y_range * 0.12
            plot_min_y = min_y - pad_y
            plot_max_y = max_y + pad_y

            m_left, m_right, m_top, m_bottom = 65, 35, 40, 45
            plot_w = w - m_left - m_right
            plot_h = h - m_top - m_bottom
            total_n = len(all_vals)

            def to_canvas(idx: int, val: float) -> Tuple[float, float]:
                cx = m_left + (idx / max(1, total_n - 1)) * plot_w
                cy = m_top + plot_h - ((val - plot_min_y) / (plot_max_y - plot_min_y)) * plot_h
                return cx, cy

            # Horizontal Grid Lines & Ticks
            num_y_ticks = 5
            for i in range(num_y_ticks + 1):
                tick_val = plot_min_y + (i / num_y_ticks) * (plot_max_y - plot_min_y)
                _, cy = to_canvas(0, tick_val)
                self.create_line(m_left, cy, w - m_right, cy, fill="#232838", width=1, dash=(3, 3), tags="static_plot")
                self.create_text(m_left - 10, cy, text=f"{tick_val:.2f}", fill="#94a3b8", anchor="e", font=("Segoe UI", 8), tags="static_plot")

            # Vertical Step Ticks
            step_stride = max(1, total_n // 8)
            for i in range(0, total_n, step_stride):
                cx, _ = to_canvas(i, min_y)
                self.create_line(cx, m_top, cx, m_top + plot_h, fill="#232838", width=1, dash=(3, 3), tags="static_plot")
                lbl = f"t+{i - len(self.history) + 1}" if i >= len(self.history) else f"{i}"
                self.create_text(cx, m_top + plot_h + 12, text=lbl, fill="#94a3b8", anchor="n", font=("Segoe UI", 8), tags="static_plot")

            # Axis Base Lines
            self.create_line(m_left, m_top, m_left, m_top + plot_h, fill="#334155", width=1, tags="static_plot")
            self.create_line(m_left, m_top + plot_h, w - m_right, m_top + plot_h, fill="#334155", width=1, tags="static_plot")

            # Area Fill Under Historical Line
            base_y = m_top + plot_h
            poly_points = [m_left, base_y]
            hist_coords = []
            for i, val in enumerate(self.history):
                cx, cy = to_canvas(i, val)
                poly_points.extend([cx, cy])
                hist_coords.append((cx, cy))
            poly_points.extend([hist_coords[-1][0], base_y])

            if len(self.history) >= 2:
                self.create_polygon(*poly_points, fill="#172554", outline="", tags="static_plot")

            # Historical Line
            line_flat = []
            for cx, cy in hist_coords:
                line_flat.extend([cx, cy])
            if len(line_flat) >= 4:
                self.create_line(*line_flat, fill="#38bdf8", width=2, smooth=True, tags="static_plot")

            # Historical Points
            for i, (cx, cy) in enumerate(hist_coords):
                r = 3 if total_n < 60 else 2
                fill_color = "#f59e0b" if i == len(self.history) - 1 else "#38bdf8"
                self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill_color, outline="#0f172a", width=1, tags="static_plot")

            # Forecast Trajectory & Predictions (Marked with tags="static_plot" to remain permanent!)
            if self.forecast:
                last_hist_cx, last_hist_cy = hist_coords[-1]
                prev_cx, prev_cy = last_hist_cx, last_hist_cy

                for f_idx, f_val in enumerate(self.forecast):
                    idx = len(self.history) + f_idx
                    fcx, fcy = to_canvas(idx, f_val)

                    # Dashed connection line
                    self.create_line(prev_cx, prev_cy, fcx, fcy, fill="#10b981", width=2, dash=(4, 3), tags="static_plot")

                    # Glowing outer pulse ring
                    self.create_oval(fcx - 9, fcy - 9, fcx + 9, fcy + 9, outline="#059669", width=1, tags="static_plot")
                    # Solid inner target marker
                    self.create_oval(fcx - 5, fcy - 5, fcx + 5, fcy + 5, fill="#10b981", outline="#ffffff", width=1.5, tags="static_plot")

                    # Forecast Callout Badge
                    callout_text = f"★ Next: {f_val:.4f}"
                    self._draw_tag(fcx, fcy - 18, callout_text, bg="#065f46", fg="#a7f3d0", tags="static_plot")
                    prev_cx, prev_cy = fcx, fcy

            # Chart Title & Legend (Permanent tags="static_plot")
            self.create_text(m_left, 18, text="Historical Time Series vs. RNN Forecast", fill="#f8fafc", anchor="w", font=("Segoe UI", 11, "bold"), tags="static_plot")

            leg_x = w - m_right
            self.create_text(leg_x, 18, text="★ Forecast", fill="#10b981", anchor="e", font=("Segoe UI", 9, "bold"), tags="static_plot")
            self.create_text(leg_x - 75, 18, text="● Historical", fill="#38bdf8", anchor="e", font=("Segoe UI", 9, "bold"), tags="static_plot")

        def _draw_loss_curve(self, w: int, h: int):
            if not self.loss_data:
                self.create_text(w // 2, h // 2, text="No training loss history available yet", fill="#64748b", font=("Segoe UI", 12), tags="static_plot")
                return

            epochs = [ep for ep, _ in self.loss_data]
            losses = [ls for _, ls in self.loss_data]
            min_loss, max_loss = min(losses), max(losses)
            max_ep = max(epochs)

            m_left, m_right, m_top, m_bottom = 70, 35, 40, 45
            plot_w = w - m_left - m_right
            plot_h = h - m_top - m_bottom

            l_range = max_loss - min_loss if max_loss != min_loss else 1.0
            pad_l = l_range * 0.1
            plot_min_l = max(0.0, min_loss - pad_l)
            plot_max_l = max_loss + pad_l

            def to_canvas(ep: int, loss_val: float) -> Tuple[float, float]:
                cx = m_left + ((ep - 1) / max(1, max_ep - 1)) * plot_w
                cy = m_top + plot_h - ((loss_val - plot_min_l) / (plot_max_l - plot_min_l)) * plot_h
                return cx, cy

            # Grid & Ticks
            for i in range(5):
                tick_val = plot_min_l + (i / 4.0) * (plot_max_l - plot_min_l)
                _, cy = to_canvas(1, tick_val)
                self.create_line(m_left, cy, w - m_right, cy, fill="#232838", width=1, dash=(3, 3), tags="static_plot")
                self.create_text(m_left - 10, cy, text=f"{tick_val:.6f}", fill="#94a3b8", anchor="e", font=("Segoe UI", 8), tags="static_plot")

            # Epoch Ticks
            ep_stride = max(1, max_ep // 5)
            for ep in range(0, max_ep + 1, ep_stride):
                if ep == 0: ep = 1
                cx, _ = to_canvas(ep, min_loss)
                self.create_line(cx, m_top, cx, m_top + plot_h, fill="#232838", width=1, dash=(3, 3), tags="static_plot")
                self.create_text(cx, m_top + plot_h + 12, text=f"{ep}", fill="#94a3b8", anchor="n", font=("Segoe UI", 8), tags="static_plot")

            # Axes
            self.create_line(m_left, m_top, m_left, m_top + plot_h, fill="#334155", width=1, tags="static_plot")
            self.create_line(m_left, m_top + plot_h, w - m_right, m_top + plot_h, fill="#334155", width=1, tags="static_plot")

            # Draw Loss Line
            pts = []
            for ep, ls in self.loss_data:
                cx, cy = to_canvas(ep, ls)
                pts.extend([cx, cy])
            if len(pts) >= 4:
                self.create_line(*pts, fill="#f43f5e", width=2.5, smooth=True, tags="static_plot")

            # Title & End Points
            end_cx, end_cy = to_canvas(epochs[-1], losses[-1])
            self.create_oval(end_cx - 4, end_cy - 4, end_cx + 4, end_cy + 4, fill="#10b981", outline="#ffffff", width=1.5, tags="static_plot")

            self.create_text(m_left, 18, text=f"Training Loss Convergence (Final MSE: {losses[-1]:.6f})", fill="#f8fafc", anchor="w", font=("Segoe UI", 11, "bold"), tags="static_plot")
            self.create_text(w - m_right, 18, text=f"1000 Epochs", fill="#94a3b8", anchor="e", font=("Segoe UI", 9), tags="static_plot")

        def _draw_tag(self, x: float, y: float, text: str, bg="#1e293b", fg="#ffffff", tags="hover_tool"):
            pad_x, pad_y = 6, 3
            font = ("Segoe UI", 8, "bold")
            char_w = 6
            t_w = len(text) * char_w
            self.create_rectangle(x - t_w // 2 - pad_x, y - 9 - pad_y, x + t_w // 2 + pad_x, y + 9 + pad_y, fill=bg, outline="#334155", width=1, tags=tags)
            self.create_text(x, y, text=text, fill=fg, font=font, tags=tags)

        def _on_mouse_move(self, event):
            """
            Safely renders mouse crosshair & tooltips without touching or deleting
            any historical or forecast static plot elements.
            """
            if self.mode != "time_series" or not self.history:
                return
            all_vals = self.history + self.forecast
            total_n = len(all_vals)
            w = self.winfo_width()
            h = self.winfo_height()
            if w < 100:
                try:
                    w = int(self.cget("width"))
                except Exception:
                    pass
            if h < 80:
                try:
                    h = int(self.cget("height"))
                except Exception:
                    pass
            m_left, m_right = 65, 35
            plot_w = w - m_left - m_right

            if event.x < m_left or event.x > w - m_right:
                self.delete("hover_tool")
                return

            ratio = (event.x - m_left) / max(1, plot_w)
            idx = round(ratio * (total_n - 1))
            idx = max(0, min(total_n - 1, idx))

            val = all_vals[idx]
            min_y, max_y = min(all_vals), max(all_vals)
            y_range = max_y - min_y if max_y != min_y else 1.0
            pad_y = y_range * 0.12
            plot_min_y, plot_max_y = min_y - pad_y, max_y + pad_y
            plot_h = h - 40 - 45

            cx = m_left + (idx / max(1, total_n - 1)) * plot_w
            cy = 40 + plot_h - ((val - plot_min_y) / (plot_max_y - plot_min_y)) * plot_h

            # Delete ONLY previous hover elements
            self.delete("hover_tool")

            # Crosshair vertical line
            self.create_line(cx, 40, cx, 40 + plot_h, fill="#64748b", width=1, dash=(2, 2), tags="hover_tool")
            # Hover dot marker
            self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#38bdf8", outline="#ffffff", width=1.5, tags="hover_tool")

            is_forecast = idx >= len(self.history)
            status_txt = "Forecast" if is_forecast else f"Step {idx}"
            tip_text = f"[{status_txt}]  Value: {val:.4f}"
            self._draw_tag(cx, cy - 20, tip_text, bg="#0f172a", fg="#38bdf8", tags="hover_tool")

        def _on_mouse_leave(self, event):
            self.delete("hover_tool")


# ==============================================================================
# Professional Tkinter GUI Main Application
# ==============================================================================

if TKINTER_AVAILABLE:
    class NeuroCastGUI:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("NeuroCast — Time Series RNN Predictor | Concept & Design by K.S.Chandra 2026")
            self.root.geometry("1140x780")
            self.root.minsize(940, 660)
            self.root.configure(bg="#0b0f19")

            # State variables
            self.current_series: List[float] = []
            self.current_result: Optional[PredictionResult] = None
            self.loaded_filepath = ""
            self.available_columns: Dict[str, List[float]] = {}
            self.training_thread: Optional[threading.Thread] = None
            self.cancel_event = threading.Event()

            # Tkinter Variables
            self.var_source_type = tk.StringVar(value="preset")
            self.var_preset = tk.StringVar(value="sine_trend")
            self.var_column = tk.StringVar()
            self.var_epochs = tk.IntVar(value=1000)  # Default 1000 as requested
            self.var_lr = tk.DoubleVar(value=0.01)
            self.var_hidden = tk.IntVar(value=32)
            self.var_horizon = tk.IntVar(value=1)
            self.var_engine = tk.StringVar(value="auto")

            self._setup_modern_styles()
            self._build_layout()
            self._load_preset_data()

        def _setup_modern_styles(self):
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass

            style.configure(".", background="#0b0f19", foreground="#f8fafc", font=("Segoe UI", 9))
            style.configure("TFrame", background="#0b0f19")
            style.configure("Card.TFrame", background="#151b2b", relief="flat")
            style.configure("Header.TFrame", background="#111827")

            # Labels
            style.configure("TLabel", background="#151b2b", foreground="#f8fafc")
            style.configure("Header.TLabel", background="#111827", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
            style.configure("Muted.TLabel", background="#151b2b", foreground="#94a3b8", font=("Segoe UI", 8))
            style.configure("Section.TLabel", background="#151b2b", foreground="#38bdf8", font=("Segoe UI", 10, "bold"))

            # Buttons
            style.configure("Accent.TButton", background="#2563eb", foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=0, padding=8)
            style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#334155")])

            style.configure("Primary.TButton", background="#0284c7", foreground="#ffffff", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=6)
            style.map("Primary.TButton", background=[("active", "#0369a1")])

            style.configure("Danger.TButton", background="#dc2626", foreground="#ffffff", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=6)
            style.map("Danger.TButton", background=[("active", "#b91c1c"), ("disabled", "#334155")])

            style.configure("Secondary.TButton", background="#334155", foreground="#ffffff", font=("Segoe UI", 9), borderwidth=0, padding=6)
            style.map("Secondary.TButton", background=[("active", "#475569")])

            # Notebook Tabs
            style.configure("TNotebook", background="#0b0f19", borderwidth=0)
            style.configure("TNotebook.Tab", background="#1e293b", foreground="#94a3b8", padding=[16, 7], font=("Segoe UI", 9, "bold"), borderwidth=0)
            style.map("TNotebook.Tab", background=[("selected", "#2563eb")], foreground=[("selected", "#ffffff")])

            # Progress Bar
            style.configure("Horizontal.TProgressbar", background="#10b981", troughcolor="#1e293b", borderwidth=0)

            # Combobox styling to ensure dark background and high visibility
            style.configure("TCombobox", fieldbackground="#1e293b", background="#334155", foreground="#f8fafc", arrowcolor="#38bdf8")
            style.map("TCombobox",
                fieldbackground=[("readonly", "#1e293b")],
                selectbackground=[("readonly", "#2563eb")],
                selectforeground=[("readonly", "#ffffff")],
                foreground=[("readonly", "#f8fafc")]
            )

        def _build_layout(self):
            # Top App Bar
            header = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 10))
            header.pack(fill="x", side="top")

            title_box = ttk.Frame(header, style="Header.TFrame")
            title_box.pack(side="left")

            ttk.Label(title_box, text="⚡ NeuroCast", font=("Segoe UI", 14, "bold"), foreground="#38bdf8", style="Header.TLabel").pack(side="left")
            ttk.Label(title_box, text="  |  Time Series RNN Forecaster", font=("Segoe UI", 10), foreground="#94a3b8", style="Header.TLabel").pack(side="left")
            ttk.Label(title_box, text="   •   Concept & Design by K.S.Chandra 2026", font=("Segoe UI", 9, "italic"), foreground="#38bdf8", style="Header.TLabel").pack(side="left")

            engine_badge_text = "NumPy Engine (Fast)" if NUMPY_AVAILABLE else ("PyTorch Engine" if TORCH_AVAILABLE else "Pure Python (Zero-Dep)")
            badge = tk.Label(header, text=f"Active: {engine_badge_text}", bg="#1e293b", fg="#10b981", font=("Segoe UI", 8, "bold"), padx=10, pady=4)
            badge.pack(side="right")

            # Dedicated Footer Bar across the bottom
            footer = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 6))
            footer.pack(fill="x", side="bottom")

            ttk.Label(footer, text="Concept & Design by K.S.Chandra 2026", font=("Segoe UI", 8, "bold"), foreground="#38bdf8", style="Header.TLabel").pack(side="left")
            ttk.Label(footer, text="NeuroCast Deep Sequence Engine • 1000 Epochs Default • Cross-Platform", font=("Segoe UI", 8), foreground="#64748b", style="Header.TLabel").pack(side="right")

            # Main Body Container
            main_container = ttk.Frame(self.root, padding=12)
            main_container.pack(fill="both", expand=True)

            # Left Sidebar (Data & Hyperparameters)
            left_pane = ttk.Frame(main_container, width=390)
            left_pane.pack(side="left", fill="y", padx=(0, 10))
            left_pane.pack_propagate(False)

            # Right Panel (Visualizations & Results)
            right_pane = ttk.Frame(main_container)
            right_pane.pack(side="right", fill="both", expand=True)

            self._build_data_source_card(left_pane)
            self._build_hyperparams_card(left_pane)
            self._build_action_card(left_pane)

            self._build_training_status_bar(right_pane)
            self._build_viz_tabs(right_pane)
            self._build_results_summary_card(right_pane)

        # ----------------------------------------------------------------------
        # Left Panel Components
        # ----------------------------------------------------------------------
        def _build_data_source_card(self, parent):
            card = ttk.Frame(parent, style="Card.TFrame", padding=14)
            card.pack(fill="x", pady=(0, 10))

            ttk.Label(card, text="1. DATA SOURCE SELECTION", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

            # Custom High-Contrast Segmented Button Bar (Fixes white color hiding options)
            source_options = [
                ("preset", "⚡ Presets"),
                ("file", "📁 Browse File"),
                ("manual", "✍️ Manual")
            ]
            self.seg_source = SegmentedButtonBar(
                card, options=source_options,
                on_change=lambda val: self._on_segmented_source_changed(val)
            )
            self.seg_source.pack(fill="x", pady=(0, 10))

            # Container for dynamic sub-options
            self.source_content_box = ttk.Frame(card, style="Card.TFrame")
            self.source_content_box.pack(fill="x", pady=(0, 6))

            # Data Preview Stats Box
            self.stats_box = tk.Label(
                card, text="Loaded: 0 points | Min: - | Max: - | Mean: -",
                bg="#1e293b", fg="#94a3b8", font=("Consolas", 8), anchor="w", padx=8, pady=6, justify="left"
            )
            self.stats_box.pack(fill="x", pady=(6, 0))

            # Initial sub-view population
            self._render_source_subview("preset")

        def _on_segmented_source_changed(self, val: str):
            self.var_source_type.set(val)
            self._render_source_subview(val)

        def _render_source_subview(self, mode: str):
            for widget in self.source_content_box.winfo_children():
                widget.destroy()

            if mode == "preset":
                ttk.Label(self.source_content_box, text="Choose Synthetic Wave Pattern:", font=("Segoe UI", 8, "bold"), foreground="#38bdf8").pack(anchor="w", pady=(2, 4))
                presets = [
                    ("sine_trend", "Sine Wave + Linear Trend (50 pts)"),
                    ("stock_walk", "Asset Price Random Walk (50 pts)"),
                    ("seasonal_sales", "Multi-frequency Seasonal Sales (50 pts)"),
                    ("damped_oscillation", "Damped Physical Harmonic Wave (50 pts)")
                ]
                cb = ttk.Combobox(self.source_content_box, textvariable=self.var_preset, values=[k for k, _ in presets], state="readonly")
                cb.pack(fill="x", pady=(0, 4))
                cb.bind("<<ComboboxSelected>>", lambda e: self._load_preset_data())
                if hasattr(self, "chart_forecast"):
                    self._load_preset_data()

            elif mode == "file":
                ttk.Label(self.source_content_box, text="Load Data File (CSV, JSON, TXT, Excel):", font=("Segoe UI", 8, "bold"), foreground="#38bdf8").pack(anchor="w", pady=(2, 4))
                f_box = ttk.Frame(self.source_content_box, style="Card.TFrame")
                f_box.pack(fill="x", pady=(0, 4))
                btn_browse = ttk.Button(f_box, text="Browse File...", style="Primary.TButton", command=self._open_file_dialog)
                btn_browse.pack(side="left", fill="x", expand=True, padx=(0, 4))

                local_files = find_workspace_data_files()
                if local_files:
                    f_names = [os.path.basename(f) for f in local_files]
                    self.cb_detected = ttk.Combobox(f_box, values=f_names, state="readonly", width=14)
                    self.cb_detected.set(f_names[0])
                    self.cb_detected.pack(side="right")
                    self.cb_detected.bind("<<ComboboxSelected>>", lambda e: self._load_file_from_path(local_files[self.cb_detected.current()]))

                self.col_select_frame = ttk.Frame(self.source_content_box, style="Card.TFrame")
                self.col_select_frame.pack(fill="x")

                if local_files:
                    self._load_file_from_path(local_files[0])

            elif mode == "manual":
                ttk.Label(self.source_content_box, text="Enter Sequence (Comma/Space Separated):", font=("Segoe UI", 8, "bold"), foreground="#38bdf8").pack(anchor="w", pady=(2, 4))
                self.entry_manual = tk.Entry(
                    self.source_content_box, bg="#1e293b", fg="#38bdf8",
                    insertbackground="#38bdf8", font=("Consolas", 9), relief="flat", bd=4
                )
                self.entry_manual.insert(0, "10.5, 11.2, 12.0, 13.5, 14.8, 16.2, 17.5, 19.0, 20.4")
                self.entry_manual.pack(fill="x", pady=(0, 4))
                btn_apply = ttk.Button(self.source_content_box, text="Apply Manual Values", style="Primary.TButton", command=self._load_manual_data)
                btn_apply.pack(fill="x")
                self._load_manual_data()

        def _build_hyperparams_card(self, parent):
            card = ttk.Frame(parent, style="Card.TFrame", padding=14)
            card.pack(fill="x", pady=(0, 10))

            ttk.Label(card, text="2. MODEL HYPERPARAMETERS", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

            grid = ttk.Frame(card, style="Card.TFrame")
            grid.pack(fill="x")

            # Epochs (Default 1000)
            ttk.Label(grid, text="Training Epochs:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=4)
            ep_box = ttk.Frame(grid, style="Card.TFrame")
            ep_box.grid(row=0, column=1, sticky="e", pady=4)
            tk.Spinbox(ep_box, from_=10, to=10000, increment=100, textvariable=self.var_epochs, width=8, bg="#1e293b", fg="#ffffff", insertbackground="#ffffff").pack(side="left")
            for ep_val in [500, 1000]:
                tk.Button(
                    ep_box, text=str(ep_val), font=("Segoe UI", 7), bg="#334155", fg="#ffffff", relief="flat", padx=3,
                    command=lambda v=ep_val: self.var_epochs.set(v)
                ).pack(side="left", padx=(2, 0))

            # Learning Rate
            ttk.Label(grid, text="Learning Rate (Adam):", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=4)
            tk.Spinbox(grid, from_=0.001, to=0.5, increment=0.005, format="%.3f", textvariable=self.var_lr, width=8, bg="#1e293b", fg="#ffffff", insertbackground="#ffffff").grid(row=1, column=1, sticky="e", pady=4)

            # Hidden Neurons
            ttk.Label(grid, text="Hidden Units:", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", pady=4)
            ttk.Combobox(grid, textvariable=self.var_hidden, values=[16, 32, 64, 128], width=7, state="readonly").grid(row=2, column=1, sticky="e", pady=4)

            # Forecast Horizon (steps)
            ttk.Label(grid, text="Forecast Horizon:", font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w", pady=4)
            ttk.Combobox(grid, textvariable=self.var_horizon, values=[1, 3, 5], width=7, state="readonly").grid(row=3, column=1, sticky="e", pady=4)

            # Computation Engine
            ttk.Label(grid, text="Execution Engine:", font=("Segoe UI", 9)).grid(row=4, column=0, sticky="w", pady=4)
            engine_choices = ["Auto"]
            if NUMPY_AVAILABLE: engine_choices.append("NumPy")
            if TORCH_AVAILABLE: engine_choices.append("PyTorch")
            engine_choices.append("Pure Python")
            ttk.Combobox(grid, textvariable=self.var_engine, values=engine_choices, width=10, state="readonly").grid(row=4, column=1, sticky="e", pady=4)

        def _build_action_card(self, parent):
            card = ttk.Frame(parent, style="Card.TFrame", padding=14)
            card.pack(fill="x")

            self.btn_train = ttk.Button(card, text="▶  Train Model & Forecast", style="Accent.TButton", command=self._start_training_thread)
            self.btn_train.pack(fill="x", pady=(0, 6))

            btn_row = ttk.Frame(card, style="Card.TFrame")
            btn_row.pack(fill="x")

            self.btn_stop = ttk.Button(btn_row, text="⏹ Stop", style="Danger.TButton", command=self._cancel_training, state="disabled")
            self.btn_stop.pack(side="left", expand=True, fill="x", padx=(0, 4))

            btn_reset = ttk.Button(btn_row, text="↺ Reset", style="Secondary.TButton", command=self._reset_defaults)
            btn_reset.pack(side="right", expand=True, fill="x")

        # ----------------------------------------------------------------------
        # Right Panel Components
        # ----------------------------------------------------------------------
        def _build_training_status_bar(self, parent):
            card = ttk.Frame(parent, style="Card.TFrame", padding=10)
            card.pack(fill="x", pady=(0, 8))

            top_row = ttk.Frame(card, style="Card.TFrame")
            top_row.pack(fill="x", pady=(0, 4))

            self.lbl_progress = ttk.Label(top_row, text="Ready to train (Default: 1000 epochs)", font=("Segoe UI", 9, "bold"))
            self.lbl_progress.pack(side="left")

            self.lbl_metrics = ttk.Label(top_row, text="Epoch 0 / 1000 | Loss: - | Elapsed: 0.0s", font=("Consolas", 8), foreground="#94a3b8")
            self.lbl_metrics.pack(side="right")

            self.prog_bar = ttk.Progressbar(card, style="Horizontal.TProgressbar", maximum=1000, value=0)
            self.prog_bar.pack(fill="x")

        def _build_viz_tabs(self, parent):
            self.notebook = ttk.Notebook(parent)
            self.notebook.pack(fill="both", expand=True, pady=(0, 8))

            # Tab 1: Time Series & Forecast Plot
            tab_forecast = ttk.Frame(self.notebook, style="Card.TFrame")
            self.notebook.add(tab_forecast, text="  📈 Sequence & Forecast  ")
            self.chart_forecast = CanvasPlot(tab_forecast, bg="#111827")
            self.chart_forecast.pack(fill="both", expand=True)

            # Tab 2: Loss Convergence Curve
            tab_loss = ttk.Frame(self.notebook, style="Card.TFrame")
            self.notebook.add(tab_loss, text="  📉 Loss Curve (1000 Epochs)  ")
            self.chart_loss = CanvasPlot(tab_loss, bg="#111827")
            self.chart_loss.pack(fill="both", expand=True)

            # Tab 3: Historical Data & Execution Log
            tab_log = ttk.Frame(self.notebook, style="Card.TFrame", padding=8)
            self.notebook.add(tab_log, text="  📋 Data & Console Log  ")
            self.txt_log = tk.Text(tab_log, bg="#111827", fg="#e2e8f0", insertbackground="#ffffff", font=("Consolas", 9), relief="flat")
            scroll = ttk.Scrollbar(tab_log, orient="vertical", command=self.txt_log.yview)
            self.txt_log.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            self.txt_log.pack(side="left", fill="both", expand=True)

        def _build_results_summary_card(self, parent):
            card = ttk.Frame(parent, style="Card.TFrame", padding=12)
            card.pack(fill="x")

            tile_grid = ttk.Frame(card, style="Card.TFrame")
            tile_grid.pack(fill="x", pady=(0, 8))

            for col in range(4):
                tile_grid.columnconfigure(col, weight=1)

            self.tile_last = self._create_metric_tile(tile_grid, 0, "LAST OBSERVED", "-", "#94a3b8")
            self.tile_pred = self._create_metric_tile(tile_grid, 1, "FORECAST VALUE", "-", "#38bdf8")
            self.tile_diff = self._create_metric_tile(tile_grid, 2, "PROJECTED DELTA", "-", "#10b981")
            self.tile_loss = self._create_metric_tile(tile_grid, 3, "FINAL MSE LOSS", "-", "#f59e0b")

            action_bar = ttk.Frame(card, style="Card.TFrame")
            action_bar.pack(fill="x")

            ttk.Label(action_bar, text="Concept & Design by K.S.Chandra 2026", font=("Segoe UI", 8, "italic"), foreground="#64748b").pack(side="left")

            btn_copy = ttk.Button(action_bar, text="📋 Copy Report", style="Secondary.TButton", command=self._copy_results)
            btn_copy.pack(side="right", padx=(4, 0))

            btn_export = ttk.Button(action_bar, text="💾 Save Forecast (CSV)", style="Secondary.TButton", command=self._export_results)
            btn_export.pack(side="right")

        def _create_metric_tile(self, parent, col: int, header: str, default_val: str, accent: str):
            box = tk.Frame(parent, bg="#1e293b", padx=8, pady=6)
            box.grid(row=0, column=col, sticky="nsew", padx=3)
            tk.Label(box, text=header, bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 7, "bold")).pack(anchor="w")
            lbl_val = tk.Label(box, text=default_val, bg="#1e293b", fg=accent, font=("Segoe UI", 12, "bold"))
            lbl_val.pack(anchor="w")
            return lbl_val

        # ----------------------------------------------------------------------
        # Data Loading & Event Handlers
        # ----------------------------------------------------------------------
        def _load_preset_data(self):
            preset_name = self.var_preset.get()
            self.current_series = generate_synthetic_series(preset_name, n_steps=50)
            self._update_stats_display(f"Preset: {preset_name}")
            self.chart_forecast.set_time_series(self.current_series)

        def _open_file_dialog(self):
            path = filedialog.askopenfilename(
                title="Select Time Series Data File",
                filetypes=[
                    ("All Supported Files", "*.csv;*.tsv;*.json;*.txt;*.xlsx;*.xls"),
                    ("CSV Files (*.csv)", "*.csv"),
                    ("JSON Files (*.json)", "*.json"),
                    ("Text Files (*.txt)", "*.txt"),
                    ("All Files (*.*)", "*.*")
                ]
            )
            if path:
                self._load_file_from_path(path)

        def _load_file_from_path(self, filepath: str):
            try:
                self.loaded_filepath = filepath
                cols = inspect_file_columns(filepath)
                self.available_columns = cols

                for w in self.col_select_frame.winfo_children():
                    w.destroy()

                col_names = list(cols.keys())
                if not col_names:
                    raise ValueError("No numeric data columns found in file.")

                ttk.Label(self.col_select_frame, text="Select Data Column:", font=("Segoe UI", 8, "bold"), foreground="#38bdf8").pack(anchor="w", pady=(2, 2))
                cb_cols = ttk.Combobox(self.col_select_frame, textvariable=self.var_column, values=col_names, state="readonly")
                cb_cols.set(col_names[0])
                cb_cols.pack(fill="x")
                cb_cols.bind("<<ComboboxSelected>>", lambda e: self._on_column_selected())

                self._on_column_selected()
            except Exception as e:
                messagebox.showerror("File Error", f"Could not load data file:\n{e}")

        def _on_column_selected(self):
            col = self.var_column.get()
            if col in self.available_columns:
                self.current_series = self.available_columns[col]
                fname = os.path.basename(self.loaded_filepath) if self.loaded_filepath else "File"
                self._update_stats_display(f"{fname} [{col}]")
                self.chart_forecast.set_time_series(self.current_series)

        def _load_manual_data(self):
            raw = self.entry_manual.get()
            nums = []
            for token in raw.replace(",", " ").split():
                try:
                    nums.append(float(token))
                except ValueError:
                    pass
            if len(nums) < 2:
                messagebox.showwarning("Data Warning", "Please enter at least 2 valid numeric points.")
                return
            self.current_series = nums
            self._update_stats_display("Manual Input")
            self.chart_forecast.set_time_series(self.current_series)

        def _update_stats_display(self, source_name: str):
            n = len(self.current_series)
            if n > 0:
                min_v = min(self.current_series)
                max_v = max(self.current_series)
                mean_v = sum(self.current_series) / n
                last_v = self.current_series[-1]
                if hasattr(self, "stats_box"):
                    self.stats_box.configure(
                        text=f"Source : {source_name}\nCount  : {n} points\nMin    : {min_v:.2f} | Max: {max_v:.2f}\nMean   : {mean_v:.2f} | Last: {last_v:.4f}"
                    )
                if hasattr(self, "tile_last"):
                    self.tile_last.configure(text=f"{last_v:.4f}")
            else:
                if hasattr(self, "stats_box"):
                    self.stats_box.configure(text="No data points loaded")

        # ----------------------------------------------------------------------
        # Background Training Workflow
        # ----------------------------------------------------------------------
        def _start_training_thread(self):
            if not self.current_series or len(self.current_series) < 2:
                messagebox.showerror("Error", "Please load at least 2 time-series observations before training.")
                return

            if self.training_thread and self.training_thread.is_alive():
                return

            self.cancel_event.clear()
            epochs = self.var_epochs.get()
            self.prog_bar.configure(maximum=epochs, value=0)
            self.btn_train.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.lbl_progress.configure(text=f"Training RNN across {epochs} epochs...")

            self.txt_log.delete("1.0", "end")
            self._log_msg(f"=== NeuroCast RNN Training Started ===")
            self._log_msg(f"Concept & Design by K.S.Chandra 2026")
            self._log_msg(f"Target Observations: {len(self.current_series)}")
            self._log_msg(f"Configured Epochs : {epochs}")
            self._log_msg(f"Learning Rate     : {self.var_lr.get()}")
            self._log_msg(f"Hidden Neurons    : {self.var_hidden.get()}")
            self._log_msg(f"Engine Selection  : {self.var_engine.get()}")
            self._log_msg("-" * 50)

            self.training_thread = threading.Thread(target=self._run_training_worker, daemon=True)
            self.training_thread.start()

        def _run_training_worker(self):
            try:
                res = train_rnn_engine(
                    data=self.current_series,
                    num_epochs=self.var_epochs.get(),
                    learning_rate=self.var_lr.get(),
                    hidden_size=self.var_hidden.get(),
                    forecast_steps=self.var_horizon.get(),
                    engine_preference=self.var_engine.get(),
                    progress_callback=self._on_training_progress,
                    cancel_event=self.cancel_event
                )
                self.root.after(0, lambda: self._on_training_finished(res))
            except Exception as e:
                err_text = traceback.format_exc()
                self.root.after(0, lambda: self._on_training_error(str(e), err_text))

        def _on_training_progress(self, epoch: int, total_epochs: int, loss: float, elapsed: float):
            self.root.after(0, lambda: self._update_progress_ui(epoch, total_epochs, loss, elapsed))

        def _update_progress_ui(self, epoch: int, total_epochs: int, loss: float, elapsed: float):
            self.prog_bar["value"] = epoch
            pct = (epoch / total_epochs) * 100.0
            speed = (epoch / elapsed) if elapsed > 0 else 0
            self.lbl_metrics.configure(text=f"Epoch {epoch}/{total_epochs} ({pct:.0f}%) | Loss: {loss:.6f} | {speed:.1f} ep/s | {elapsed:.1f}s")
            if epoch % max(1, total_epochs // 10) == 0:
                self._log_msg(f"Epoch [{epoch:>{len(str(total_epochs))}}/{total_epochs}]  |  Average Loss: {loss:.6f}")

        def _on_training_finished(self, result: PredictionResult):
            self.current_result = result
            self.btn_train.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.lbl_progress.configure(text=f"Training Complete ({result.epochs_completed} epochs)")

            self.chart_forecast.set_time_series(result.history, result.predictions)
            self.chart_loss.set_loss_curve(result.loss_history)

            self.tile_last.configure(text=f"{result.last_val:.4f}")
            self.tile_pred.configure(text=f"{result.predicted_val:.4f}")

            dir_sym = result.direction_symbol
            dir_col = "#10b981" if result.diff >= 0 else "#f43f5e"
            self.tile_diff.configure(text=f"{dir_sym} {abs(result.diff):.4f} ({result.pct_change:+.2f}%)", fg=dir_col)
            self.tile_loss.configure(text=f"{result.final_loss:.6f}")

            self._log_msg("-" * 50)
            self._log_msg(f"Engine Used         : {result.engine_used}")
            self._log_msg(f"Training Time       : {result.training_time:.2f} seconds")
            self._log_msg(f"Last Observed Value : {result.last_val:.4f}")
            self._log_msg(f"Predicted Next Value: {result.predicted_val:.4f}")
            self._log_msg(f"Projected Change    : {result.direction} ({result.pct_change:+.2f}%)")
            if len(result.predictions) > 1:
                self._log_msg(f"Full Rollout Steps  : {[round(x, 4) for x in result.predictions]}")
            self._log_msg("=== Done ===")

        def _on_training_error(self, err_msg: str, traceback_str: str):
            self.btn_train.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.lbl_progress.configure(text="Training Failed")
            self._log_msg(f"\n[ERROR] Training failed:\n{traceback_str}")
            messagebox.showerror("Training Error", f"An error occurred during training:\n{err_msg}")

        def _cancel_training(self):
            self.cancel_event.set()
            self.lbl_progress.configure(text="Stopping training...")
            self.btn_stop.configure(state="disabled")

        def _reset_defaults(self):
            self.var_epochs.set(1000)
            self.var_lr.set(0.01)
            self.var_hidden.set(32)
            self.var_horizon.set(1)
            self.var_engine.set("auto")
            self.prog_bar["value"] = 0
            self.lbl_progress.configure(text="Settings reset to defaults (1000 epochs).")

        def _log_msg(self, text: str):
            self.txt_log.insert("end", text + "\n")
            self.txt_log.see("end")

        def _copy_results(self):
            if not self.current_result:
                messagebox.showinfo("Info", "No prediction results available to copy.")
                return
            res = self.current_result
            text = (
                f"==================================================\n"
                f"NeuroCast RNN Time Series Prediction Report\n"
                f"Concept & Design by K.S.Chandra 2026\n"
                f"==================================================\n"
                f"Engine Used         : {res.engine_used}\n"
                f"Epochs Completed    : {res.epochs_completed}\n"
                f"Final Training Loss : {res.final_loss:.6f}\n"
                f"Training Duration   : {res.training_time:.2f}s\n"
                f"Last Observed Value : {res.last_val:.4f}\n"
                f"Predicted Value     : {res.predicted_val:.4f}\n"
                f"Expected Delta      : {res.direction} of {abs(res.diff):.4f} ({res.pct_change:+.2f}%)\n"
                f"==================================================\n"
            )
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Forecast summary copied to clipboard!\nConcept & Design by K.S.Chandra 2026")

        def _export_results(self):
            if not self.current_result:
                messagebox.showinfo("Info", "No prediction results available to export.")
                return
            path = filedialog.asksaveasfilename(
                title="Save Prediction Results",
                defaultextension=".csv",
                filetypes=[("CSV File (*.csv)", "*.csv"), ("Text Report (*.txt)", "*.txt")]
            )
            if not path:
                return
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["# NeuroCast Prediction Report - Concept & Design by K.S.Chandra 2026"])
                    writer.writerow(["Step_Type", "Index", "Value"])
                    for i, val in enumerate(self.current_result.history):
                        writer.writerow(["Historical", i, val])
                    for f_idx, f_val in enumerate(self.current_result.predictions):
                        writer.writerow(["Forecast", len(self.current_result.history) + f_idx, f_val])
                messagebox.showinfo("Export Success", f"Forecast saved successfully to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save file:\n{e}")


# ==============================================================================
# Enhanced Interactive Console / Headless Fallback Mode
# ==============================================================================

def run_cli_interactive():
    engine_name = "PyTorch" if TORCH_AVAILABLE else ("NumPy (Accelerated)" if NUMPY_AVAILABLE else "Pure Python (Zero-Dep)")
    print("\n" + "=" * 68)
    print("      NEUROCAST: TIME SERIES RNN PREDICTOR (CLI MODE)")
    print("      Concept & Design by K.S.Chandra 2026")
    print(f"      Active Engine: {engine_name}  |  Default Epochs: 1000")
    print("=" * 68)
    print("Select data source option:")
    print("  [1] Select from Detected Workspace Files")
    print("  [2] Enter File Path Manually (CSV, JSON, TXT, Excel)")
    print("  [3] Direct Console Input (Space or comma separated)")
    print("  [4] Instant Demo: Sine Wave + Linear Trend (50 points)")
    print("  [5] Instant Demo: Asset Price Random Walk (50 points)")
    print("  [0] Exit")
    print("-" * 68)

    choice = input("Select option [1-5, or 0] (default: 4): ").strip() or "4"
    if choice == "0":
        return

    series: Optional[List[float]] = None

    if choice == "1":
        files = find_workspace_data_files()
        if not files:
            print("\n [!] No data files detected in current folder.")
            return
        print("\nDetected files:")
        for idx, f in enumerate(files):
            print(f"  [{idx}] {os.path.basename(f)}")
        pick = input(f"Select index [0-{len(files)-1}] (default: 0): ").strip() or "0"
        pick_idx = int(pick) if pick.isdigit() and int(pick) < len(files) else 0
        cols = inspect_file_columns(files[pick_idx])
        col_names = list(cols.keys())
        if len(col_names) > 1:
            print("\nAvailable columns:")
            for c_idx, c_name in enumerate(col_names):
                print(f"  [{c_idx}] {c_name} ({len(cols[c_name])} values)")
            c_pick = input(f"Select column index [0-{len(col_names)-1}] (default: 0): ").strip() or "0"
            c_idx = int(c_pick) if c_pick.isdigit() and int(c_pick) < len(col_names) else 0
            series = cols[col_names[c_idx]]
        else:
            series = cols[col_names[0]]

    elif choice == "2":
        path = input("Enter file path: ").strip().strip('"').strip("'")
        if not os.path.isfile(path):
            print(f" [!] File does not exist: {path}")
            return
        cols = inspect_file_columns(path)
        col_names = list(cols.keys())
        series = cols[col_names[0]]

    elif choice == "3":
        raw = input("Enter values separated by commas or spaces: ").strip()
        series = []
        for token in raw.replace(",", " ").split():
            try:
                series.append(float(token))
            except ValueError:
                pass

    elif choice == "4":
        series = generate_synthetic_series("sine_trend", 50)
    elif choice == "5":
        series = generate_synthetic_series("stock_walk", 50)

    if not series or len(series) < 2:
        print("\n [!] Error: Need at least 2 numeric values to train RNN.")
        return

    print("\n" + "-" * 68)
    print(f"Observations count : {len(series)}")
    print(f"First 5 values     : {series[:5]}")
    print(f"Last 5 values      : {series[-5:]}")
    print(f"Last observed point: {series[-1]}")
    print("-" * 68)

    ep_input = input("Enter number of epochs [Press Enter for default: 1000]: ").strip()
    epochs = int(ep_input) if ep_input.isdigit() and int(ep_input) > 0 else 1000

    lr_input = input("Enter learning rate   [Press Enter for default: 0.01]: ").strip()
    try:
        lr = float(lr_input) if lr_input else 0.01
    except ValueError:
        lr = 0.01

    print("\nTraining RNN model...")

    def cli_progress(ep, total_ep, loss, elapsed):
        if ep % max(1, total_ep // 10) == 0 or ep == total_ep:
            pct = (ep / total_ep) * 100
            print(f"Epoch [{ep:>{len(str(total_ep))}}/{total_ep}] ({pct:>3.0f}%)  |  Loss: {loss:.6f}  |  Elapsed: {elapsed:.2f}s")

    result = train_rnn_engine(series, num_epochs=epochs, learning_rate=lr, progress_callback=cli_progress)

    print("\n" + "=" * 68)
    print("                    PREDICTION RESULT")
    print("           Concept & Design by K.S.Chandra 2026")
    print("=" * 68)
    print(f" Compute Engine Used   : {result.engine_used}")
    print(f" Epochs Completed      : {result.epochs_completed}")
    print(f" Final Training Loss   : {result.final_loss:.6f}")
    print(f" Last Observed Value   : {result.last_val:.4f}")
    print(f" Forecast Next Value   : {result.predicted_val:.4f}")
    print(f" Projected Change      : {result.direction_symbol} {result.direction} of {abs(result.diff):.4f} ({result.pct_change:+.2f}%)")
    print(f" Total Duration        : {result.training_time:.2f}s")
    print("=" * 68 + "\n")


# ==============================================================================
# Application Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="NeuroCast: Time Series RNN Predictor - Concept & Design by K.S.Chandra 2026")
    parser.add_argument("--cli", action="store_true", help="Force interactive CLI mode instead of GUI")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
    args, unknown = parser.parse_known_args()

    use_gui = TKINTER_AVAILABLE and not args.cli

    if use_gui and (IS_LINUX or IS_MAC):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            use_gui = False

    if use_gui:
        try:
            root = tk.Tk()
            app = NeuroCastGUI(root)
            root.mainloop()
        except Exception as e:
            print(f"\n[Notice] Could not initialize graphical window ({e}). Falling back to CLI mode...")
            run_cli_interactive()
    else:
        run_cli_interactive()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print("\n" + "=" * 68)
        print("[CRITICAL ERROR] An unexpected error occurred:")
        print("=" * 68)
        traceback.print_exc()
