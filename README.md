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

## 🌐 Dual-Mode Web Deployment: GitHub Pages (Autonomous) & Cloud Servers

NeuroCast features an intelligent **Dual-Engine Architecture** that adapts automatically to its hosting environment:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HOSTING ENVIRONMENT DETECTION                         │
├──────────────────────────────────────┬──────────────────────────────────────┤
│  STATIC HOSTING (GitHub Pages, etc.) │  CLOUD SERVER (Render, Railway, etc) │
│  • Autonomous Client Engine (app.js) │  • Hidden Python Backend (server.py) │
│  • Client-Side CSV/JSON/TXT Parsing  │  • NumPy / PyTorch Acceleration      │
│  • 1000 Epochs in ~100ms in Browser  │  • Multi-threaded ThreadingHTTPServer│
│  • Zero server required / Zero 404s  │  • Full REST API (/api/train)        │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### 🐙 GitHub Pages Deployment Guide (Step-by-Step)

If you deployed to GitHub Pages and saw:
> `Error loading file: Unexpected token '<', "<html><he"... is not valid JSON`

#### Why This Occurred:
GitHub Pages is a **static web host**. It does not run Python or backend servers. When an earlier version of the frontend made a network request to `/api/inspect-file` or `/api/train`, GitHub Pages returned its default `404 Not Found` HTML page (`<html><head>...`). When the browser tried to parse this HTML page with `response.json()`, it threw the "Unexpected token '<'" error.

#### How It Is Solved:
1. **Autonomous Client Engine**: `app.js` now contains a self-contained recurrent neural network with Adam optimization and BPTT that executes 1000 epochs in ~100ms entirely in the client browser.
2. **Client-Side File Parsing**: CSV, JSON, TSV, and TXT files are read and parsed directly inside the browser using modern File APIs, eliminating all `/api/inspect-file` network requests.
3. **Repository Sync**: The web app files (`index.html`, `style.css`, `app.js`, `.nojekyll`) are synced across **Root `/`**, **`/docs`**, and **`/public`**.

#### 3-Step Setup for GitHub Pages:
1. **Commit and Push the Updated Files**:
   Ensure the following files in the project root are pushed to your GitHub repository:
   - `index.html`
   - `style.css`
   - `app.js`
   - `.nojekyll`
   - `docs/` folder (optional backup)
2. **Configure GitHub Pages**:
   - Open your GitHub repository in your browser.
   - Go to **Settings** > **Pages** (under "Code and automation" on the left).
   - Under **Build and deployment** > **Source**, choose **Deploy from a branch**.
   - Under **Branch**, select `main` (or `master`) and keep folder as `/ (root)`.
   - Click **Save**.
3. **Verify Deployment**:
   - Visit your site URL: `https://<username>.github.io/<repository-name>/`.
   - The status chip in the top right will display: `Autonomous Mode (Client Engine)`.
   - File uploads (CSV/JSON/TXT), 1000-epoch training, interactive hover plots, and report exports work instantly!

---

## ☁️ Full-Stack Cloud Server Hosting (With Hidden Backend)

If you want the Python backend to run on a dedicated server (with PyTorch or NumPy acceleration):

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
