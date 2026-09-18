#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

echo "================================================================="
echo "   VeriGraph: Enterprise Agentic GraphRAG & Verifiable Engine    "
echo "================================================================="

# Create and activate virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "[*] Setting up virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[*] Installing dependencies from requirements.txt..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Launch via universal cross-platform runner
python run.py "$@"
