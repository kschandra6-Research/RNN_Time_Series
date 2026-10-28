"""
NeuroCast: WSGI Compatibility Layer for Traditional Web Hosts
Concept & Design by K.S.Chandra 2026

Enables running with gunicorn:
    gunicorn app:application --bind 0.0.0.0:$PORT
Or directly:
    python app.py
"""

import os
import sys
import json
import mimetypes

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engine import train_rnn_engine, generate_synthetic_series, NUMPY_AVAILABLE, TORCH_AVAILABLE
from backend.parser import parse_raw_text_data, inspect_file_columns, find_workspace_data_files

PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")


def application(environ, start_response):
    """Standard WSGI entrypoint callable."""
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    headers = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type")
    ]

    if method == "OPTIONS":
        start_response("204 No Content", headers)
        return [b""]

    # Health API
    if path == "/api/health" and method == "GET":
        data = {
            "status": "online",
            "application": "NeuroCast Web Engine (WSGI)",
            "concept_and_design": "Concept & Design by K.S.Chandra 2026",
            "default_epochs": 1000,
            "engine_detected": "NumPy (Accelerated)" if NUMPY_AVAILABLE else "Pure Python (Zero-Dep)",
            "numpy_available": NUMPY_AVAILABLE,
            "torch_available": TORCH_AVAILABLE
        }
        body = json.dumps(data).encode("utf-8")
        headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        start_response("200 OK", headers)
        return [body]

    # Train API
    if path == "/api/train" and method == "POST":
        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(content_length)
            payload = json.loads(request_body.decode("utf-8"))

            series = [float(x) for x in payload.get("series", [])]
            epochs = int(payload.get("epochs", 1000))
            lr = float(payload.get("lr", 0.01))
            hidden_size = int(payload.get("hidden_size", 32))
            horizon = int(payload.get("forecast_steps", 1))
            engine = str(payload.get("engine", "auto"))

            result = train_rnn_engine(
                data=series,
                num_epochs=epochs,
                learning_rate=lr,
                hidden_size=hidden_size,
                forecast_steps=horizon,
                engine_preference=engine
            )
            body = json.dumps(result.to_dict()).encode("utf-8")
            headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            start_response("200 OK", headers)
            return [body]
        except Exception as e:
            err = json.dumps({"error": str(e), "status": 500}).encode("utf-8")
            headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(err)))])
            start_response("500 Internal Server Error", headers)
            return [err]

    # Static files serving
    rel_path = path.lstrip("/") or "index.html"
    filepath = os.path.normpath(os.path.join(PUBLIC_DIR, rel_path))
    if not filepath.startswith(PUBLIC_DIR) or not os.path.isfile(filepath):
        filepath = os.path.join(PUBLIC_DIR, "index.html")

    if os.path.isfile(filepath):
        mime, _ = mimetypes.guess_type(filepath)
        mime = mime or "text/plain"
        with open(filepath, "rb") as f:
            content = f.read()
        headers.extend([("Content-Type", f"{mime}; charset=utf-8"), ("Content-Length", str(len(content)))])
        start_response("200 OK", headers)
        return [content]

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


if __name__ == "__main__":
    from server import run_web_server
    run_web_server()
