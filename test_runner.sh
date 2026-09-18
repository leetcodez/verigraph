#!/usr/bin/env bash
set -e

# Change to script directory
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=".:$PYTHONPATH"

echo "=== Running Pytest Test Suite ==="
pytest -v tests/

echo ""
echo "=== Running Verification Benchmark ==="
python -m evaluation.run_benchmark
