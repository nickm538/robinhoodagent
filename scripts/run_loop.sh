#!/usr/bin/env bash
# Run the always-on autonomous agent. Restarts itself if it ever exits.
# Usage:  EXECUTION_MODE=paper ./scripts/run_loop.sh           (safe default)
#         EXECUTION_MODE=live LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY \
#           ROBINHOOD_MCP_TOKEN=... ./scripts/run_loop.sh --execute
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then set -a; . ./.env; set +a; fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON="${PYTHON:-python3}"

mkdir -p logs
echo "Starting rh-agent loop (mode=${EXECUTION_MODE:-paper}) — logs/loop.log"
while true; do
  "$PYTHON" -m rh_agent.cli loop "$@" 2>&1 | tee -a logs/loop.log
  echo "loop exited ($(date)) — restarting in 30s" | tee -a logs/loop.log
  sleep 30
done
