# ⚡ NeuroCast: Time Series RNN Predictor (Web & Desktop)
### *Concept & Design by K.S.Chandra 2026*

An enterprise-grade, full-stack application for interactive Time Series forecasting using Recurrent Neural Networks (RNN) and the Adam optimization algorithm.

Built with a **Zero-Hard-Dependency Policy** and **Universal Web Deployability**, NeuroCast includes a **ready-to-use modern web frontend**, a **hidden server-side neural network backend**, and native desktop GUI/CLI modes.

---

## 🌟 Architecture: Ready-to-Use Frontend & Hidden Backend

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BROWSER / CLIENT                                  │
│  • Modern Dark Slate SPA (HTML5 Canvas / CSS3 / ES6)                        │
│  • High-Contrast Segmented Buttons (No white-on-white text hiding)          │
│  • Interactive Sequence & Forecast Plot (Hover inspection with zero flicker)│
│  • Real-time Loss Convergence Curve (1000 Epochs)                           │
│  • CSV / JSON / TXT File Drag & Drop + Column Picker                        │
│  • 4-Tile Forecast Metrics Dashboard & Report Export                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST API (JSON)
                                       │ POST /api/train
                                       │ POST /api/inspect-file
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SECURE BACKEND (HIDDEN)                           │
│  • Multi-threaded Web Application Server (server.py)                        │
│  • Neural Network Weights, Gradients & Optimizer strictly on Server         │
│  • Triple Engine: PyTorch -> NumPy -> Pure Python (Zero-Dependency)         │
│  • Default 1000 Training Epochs                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Hidden Backend**: All neural network architecture, weights, forward/backward passes, gradient clipping, and optimization calculations are executed securely on the server. The user only interacts with the clean web interface and receives sanitized forecast predictions and metrics via JSON.
- **Zero-Dependency Web Server**: `server.py` uses Python's built-in `ThreadingHTTPServer` to serve the web UI and REST API without requiring any third-party framework installation.

---

## 🚀 How to Run Locally

### 1. Web Application (Recommended)
- **Windows**: Double-click `run_web.bat` or run:
  ```cmd
  run_web.bat
  ```
- **macOS / Linux**:
  ```bash
  chmod +x run_web.sh
  ./run_web.sh
  ```
- **Direct Python**:
  ```bash
  python server.py
  ```
  *Opens your browser automatically at `http://localhost:5000`.*

### 2. Desktop GUI (Tkinter)
- **Windows**: Double-click `run_predictor.bat`
- **macOS / Linux**: `./run_predictor.sh`

### 3. CLI Mode (Headless / Terminal)
```bash
python rnn_predictor.py --cli
```

---

## ☁️ Web Hosting Deployment Guide

### Option 1: Render (1-Click Blueprint)
1. Push this repository to GitHub or GitLab.
2. In [Render Dashboard](https://dashboard.render.com), click **New** > **Blueprint**.
3. Connect your repository. Render automatically reads `render.yaml` and deploys the web service.

### Option 2: Railway / Heroku
- The repository includes a `Procfile`:
  ```
  web: python server.py --no-browser
  ```
- Simply connect your repository to Railway or Heroku; it automatically binds to `$PORT` and deploys.

### Option 3: Docker (Any Cloud Host / Cloud Run / AWS / VPS)
Build and run the containerized service:
```bash
docker build -t neurocast-web .
docker run -p 5000:5000 neurocast-web
```
Or via Docker Compose:
```bash
docker compose up
```

### Option 4: Static Frontend (Vercel / Netlify / GitHub Pages)
- Deploy the files inside `public/` (`index.html`, `style.css`, `app.js`) to any static host (Vercel, Netlify).
- Set the backend API URL in `public/app.js` to point to your deployed backend.

---

## 📁 Project Structure

| Path | Purpose |
| :--- | :--- |
| `public/index.html` | Ready-to-use frontend UI with modern dark slate theme. |
| `public/style.css` | High-contrast styling (prevents white-on-white text hiding). |
| `public/app.js` | Interactive Canvas charting, hover tooltips, and API consumer. |
| `backend/engine.py` | **Hidden** RNN engines (PyTorch, NumPy, Pure Python) and Adam optimizer. |
| `backend/parser.py` | **Hidden** CSV, JSON, TXT, Excel file parsing module. |
| `server.py` | Multi-threaded Python web server and REST API handler. |
| `app.py` | WSGI entrypoint for gunicorn / traditional WSGI cloud hosts. |
| `run_web.bat` | 1-click Windows Web Launcher. |
| `run_web.sh` | 1-click macOS / Linux Web Launcher. |
| `Dockerfile` | Production container configuration. |
| `docker-compose.yml` | Multi-platform container configuration. |
| `render.yaml` | Render cloud deployment blueprint. |
| `Procfile` | Heroku / Railway process definition. |
| `rnn_predictor.py` | Desktop GUI & interactive terminal CLI application. |

---

## 👤 Credits & Attribution
**Concept & Design by K.S.Chandra 2026**
