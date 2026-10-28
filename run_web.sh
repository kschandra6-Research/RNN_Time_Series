#!/usr/bin/env bash
# ==============================================================================
# NeuroCast: Time Series RNN Predictor (Web Launcher)
# Concept & Design by K.S.Chandra 2026
# Cross-Platform Launcher for Linux and macOS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_BIN=""

if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
elif [ -x "$HOME/.local/bin/python3" ]; then
    PY_BIN="$HOME/.local/bin/python3"
elif [ -x "/usr/bin/python3" ]; then
    PY_BIN="/usr/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
    PY_BIN="/usr/local/bin/python3"
fi

if [ -z "$PY_BIN" ]; then
    echo "====================================================================="
    echo " [ERROR] Python 3 could not be found."
    echo " Please install Python 3 using your system package manager."
    echo "====================================================================="
    exit 1
fi

"$PY_BIN" "$SCRIPT_DIR/server.py" "$@"
