#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

echo "================================================================="
echo "   VeriGraph: Enterprise Agentic GraphRAG & Verifiable Engine    "
echo "================================================================="

# Activate virtualenv if present
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

export PYTHONPATH=".:$PYTHONPATH"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "[*] Initializing VeriGraph Server on http://${HOST}:${PORT}"
echo "[*] Interactive Web Dashboard:  http://localhost:${PORT}/"
echo "[*] OpenAPI Swagger Docs:      http://localhost:${PORT}/docs"
echo "================================================================="

exec uvicorn verigraph.api.app:app --host "$HOST" --port "$PORT" --reload
