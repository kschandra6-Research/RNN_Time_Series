"""
NeuroCast: Production-Ready Web Server & API
============================================
Concept & Design by K.S.Chandra 2026

Zero-dependency, multi-threaded Web Application Server.
Serves the responsive web frontend from 'public/' and powers the hidden backend RNN API.

Deployable on any cloud host (Render, Railway, Fly.io, Heroku, Docker, Cloud Run, VPS)
or runnable locally on any operating system out-of-the-box without requiring pip installs.
"""

import os
import sys
import json
import urllib.parse
import mimetypes
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engine import (
    train_rnn_engine,
    generate_synthetic_series,
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE
)
from backend.parser import (
    parse_raw_text_data,
    inspect_file_columns,
    find_workspace_data_files
)

PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server that handles concurrent requests without blocking."""
    daemon_threads = True


class NeuroCastRequestHandler(BaseHTTPRequestHandler):
    """
    Production-ready HTTP request handler.
    Serves static assets and provides REST API endpoints while hiding backend model logic.
    """

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # ----------------------------------------------------------------------
        # REST API Routes
        # ----------------------------------------------------------------------
        if path == "/api/health":
            self._send_json({
                "status": "online",
                "application": "NeuroCast Web Engine",
                "concept_and_design": "Concept & Design by K.S.Chandra 2026",
                "default_epochs": 1000,
                "engine_detected": "NumPy (Accelerated)" if NUMPY_AVAILABLE else ("PyTorch" if TORCH_AVAILABLE else "Pure Python (Zero-Dep)"),
                "numpy_available": NUMPY_AVAILABLE,
                "torch_available": TORCH_AVAILABLE
            })
            return

        elif path == "/api/presets":
            presets_data = {
                "sine_trend": {
                    "name": "Sine Wave + Linear Trend",
                    "description": "Harmonic oscillation with steady upward trajectory",
                    "series": generate_synthetic_series("sine_trend", 50)
                },
                "stock_walk": {
                    "name": "Asset Price Random Walk",
                    "description": "Simulated market financial asset series with stochastic drift",
                    "series": generate_synthetic_series("stock_walk", 50)
                },
                "seasonal_sales": {
                    "name": "Multi-frequency Seasonal Sales",
                    "description": "Dual periodic business cycles with compounding growth",
                    "series": generate_synthetic_series("seasonal_sales", 50)
                },
                "damped_oscillation": {
                    "name": "Damped Physical Harmonic Wave",
                    "description": "Decaying amplitude oscillator with restoring equilibrium",
                    "series": generate_synthetic_series("damped_oscillation", 50)
                }
            }
            self._send_json(presets_data)
            return

        elif path == "/api/sample-files":
            files = find_workspace_data_files()
            file_summaries = []
            for f in files:
                try:
                    cols = inspect_file_columns(f)
                    file_summaries.append({
                        "filename": os.path.basename(f),
                        "columns": {k: len(v) for k, v in cols.items()}
                    })
                except Exception:
                    pass
            self._send_json({"files": file_summaries})
            return

        # ----------------------------------------------------------------------
        # Static File Serving (from public/)
        # ----------------------------------------------------------------------
        rel_path = path.lstrip("/")
        if not rel_path or rel_path == "":
            rel_path = "index.html"

        safe_path = os.path.normpath(os.path.join(PUBLIC_DIR, rel_path))
        # Prevent path traversal outside PUBLIC_DIR
        if not safe_path.startswith(PUBLIC_DIR) or not os.path.isfile(safe_path):
            # Fallback to index.html for Single Page Application routing
            safe_path = os.path.join(PUBLIC_DIR, "index.html")

        if os.path.isfile(safe_path):
            self._serve_static_file(safe_path)
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        # ----------------------------------------------------------------------
        # API: /api/train (Run Hidden RNN Model & Forecast)
        # ----------------------------------------------------------------------
        if path == "/api/train":
            try:
                payload = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                self._send_error_json(400, f"Invalid JSON payload: {e}")
                return

            series = payload.get("series", [])
            if not isinstance(series, list) or len(series) < 2:
                self._send_error_json(400, "The 'series' parameter must be a list containing at least 2 numeric values.")
                return

            try:
                numeric_series = [float(x) for x in series]
            except (ValueError, TypeError):
                self._send_error_json(400, "All items in 'series' must be valid numeric values.")
                return

            epochs = int(payload.get("epochs", 1000))  # Default 1000 epochs
            lr = float(payload.get("lr", 0.01))
            hidden_size = int(payload.get("hidden_size", 32))
            forecast_steps = int(payload.get("forecast_steps", 1))
            engine_pref = str(payload.get("engine", "auto"))

            try:
                result = train_rnn_engine(
                    data=numeric_series,
                    num_epochs=epochs,
                    learning_rate=lr,
                    hidden_size=hidden_size,
                    forecast_steps=forecast_steps,
                    engine_preference=engine_pref
                )
                self._send_json(result.to_dict())
            except Exception as e:
                self._send_error_json(500, f"Training execution error: {e}")
            return

        # ----------------------------------------------------------------------
        # API: /api/inspect-file (Parse and extract columns from uploaded file)
        # ----------------------------------------------------------------------
        elif path == "/api/inspect-file":
            try:
                # Payload can be JSON with file content string or raw text
                text_content = ""
                filename = ""
                content_type = self.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    body = json.loads(post_data.decode("utf-8"))
                    text_content = body.get("content", "")
                    filename = body.get("filename", "")
                else:
                    text_content = post_data.decode("utf-8", errors="replace")

                cols = parse_raw_text_data(text_content, filename_hint=filename)
                response = {
                    "columns": {k: {"count": len(v), "sample": v[:5]} for k, v in cols.items()},
                    "data": cols
                }
                self._send_json(response)
            except Exception as e:
                self._send_error_json(400, f"Error inspecting file data: {e}")
            return

        else:
            self.send_error(404, "API Endpoint Not Found")

    def _serve_static_file(self, filepath: str):
        content_type, _ = mimetypes.guess_type(filepath)
        if not content_type:
            if filepath.endswith(".css"):
                content_type = "text/css"
            elif filepath.endswith(".js"):
                content_type = "application/javascript"
            elif filepath.endswith(".json"):
                content_type = "application/json"
            elif filepath.endswith(".svg"):
                content_type = "image/svg+xml"
            else:
                content_type = "text/plain"

        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error serving file: {e}")

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str):
        self._send_json({"error": message, "status": status}, status=status)

    def log_message(self, format, *args):
        # Clean production log formatting
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} - {args[1]} {args[2]}\n")


def run_web_server(host: str = "0.0.0.0", port: int = 5000, open_browser: bool = True):
    # Port detection from environment (supports Render, Railway, Heroku, Cloud Run)
    env_port = os.environ.get("PORT")
    if env_port and env_port.isdigit():
        port = int(env_port)

    server = ThreadedHTTPServer((host, port), NeuroCastRequestHandler)
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}" if host == "0.0.0.0" else f"http://{host}:{actual_port}"

    print("=" * 70)
    print("      ⚡ NEUROCAST: TIME SERIES RNN PREDICTOR (WEB SERVICE)")
    print("      Concept & Design by K.S.Chandra 2026")
    print(f"      Default Training Epochs: 1000  |  Backend: Secure & Hidden")
    print("=" * 70)
    print(f"  ● Web Server Active at: {url}")
    print(f"  ● Serving Frontend from: {PUBLIC_DIR}")
    print(f"  ● REST API Endpoint:    {url}/api/train")
    print(f"  ● Health Status:         {url}/api/health")
    print("=" * 70)
    print("Press Ctrl+C to terminate web server.\n")

    if open_browser:
        def _open():
            time.sleep(0.8)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down NeuroCast web service...")
        server.shutdown()
        server.server_close()
        print("[Server] Terminated cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroCast Web Server - Concept & Design by K.S.Chandra 2026")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server (default: 5000 or $PORT)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")
    args = parser.parse_args()

    run_web_server(host=args.host, port=args.port, open_browser=not args.no_browser)
