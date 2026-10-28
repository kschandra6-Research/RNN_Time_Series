"""
NeuroCast: Neural Network Engine Module
Concept & Design by K.S.Chandra 2026

Triple Engine Architecture:
1. PyTorch Engine (Accelerated GPU/CPU, if installed)
2. NumPy Engine (Vectorized matrix math, if installed)
3. Pure Python Engine (Built-in standard library, ZERO external dependencies)
"""

import math
import random
import time
import threading
from typing import List, Tuple, Optional, Callable, Dict, Any

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


class PurePythonRNNModel:
    """
    High-performance Pure Python 1-layer RNN with Adam Optimizer.
    Uses only built-in math and random modules. Zero external dependencies.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 24, output_size: int = 1, lr: float = 0.01):
        random.seed(42)
        scale = 1.0 / math.sqrt(hidden_size)
        self.H = hidden_size
        self.lr = lr

        self.Wxh = [random.uniform(-scale, scale) for _ in range(self.H)]
        self.Whh = [[random.uniform(-scale, scale) for _ in range(self.H)] for _ in range(self.H)]
        self.bh = [0.0] * self.H
        self.Why = [random.uniform(-scale, scale) for _ in range(self.H)]
        self.by = 0.0

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "predictions": self.predictions,
            "last_val": round(self.last_val, 4),
            "predicted_val": round(self.predicted_val, 4),
            "diff": round(self.diff, 4),
            "pct_change": round(self.pct_change, 2),
            "direction": self.direction,
            "direction_symbol": self.direction_symbol,
            "final_loss": float(f"{self.final_loss:.6f}"),
            "loss_history": [{"epoch": ep, "loss": float(f"{ls:.6f}")} for ep, ls in self.loss_history],
            "engine_used": self.engine_used,
            "training_time": round(self.training_time, 3),
            "epochs_completed": self.epochs_completed,
            "concept_and_design": "Concept & Design by K.S.Chandra 2026"
        }


def train_rnn_engine(data: List[float],
                     num_epochs: int = 1000,
                     learning_rate: float = 0.01,
                     hidden_size: int = 32,
                     forecast_steps: int = 1,
                     engine_preference: str = "auto",
                     progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
                     cancel_event: Optional[threading.Event] = None) -> PredictionResult:
    """
    Train RNN model for specified epochs (default 1000).
    """
    t_start = time.time()
    scaled_data, min_val, val_range = normalize_series(data)
    n_points = len(scaled_data)

    selected_engine = "pure_python"
    pref = engine_preference.lower()
    if pref == "pytorch" and TORCH_AVAILABLE:
        selected_engine = "pytorch"
    elif pref == "numpy" and NUMPY_AVAILABLE:
        selected_engine = "numpy"
    elif pref == "pure_python":
        selected_engine = "pure_python"
    else:
        if NUMPY_AVAILABLE:
            selected_engine = "numpy"
        elif TORCH_AVAILABLE:
            selected_engine = "pytorch"
        else:
            selected_engine = "pure_python"

    loss_history: List[Tuple[int, float]] = []
    final_loss = 0.0
    completed_epochs = 0

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
