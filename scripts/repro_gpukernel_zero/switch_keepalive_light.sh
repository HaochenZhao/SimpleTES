#!/usr/bin/env bash
# Search is running: drop the 144-way long decode and keep a light occupancy loop.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
PIDF="${KERNEL_LOG_DIR}/keepalive.pid"
OUT="${KERNEL_LOG_DIR}/switch_keepalive_light.out"

{
  if [[ -f "${PIDF}" ]]; then
    kill "$(cat "${PIDF}")" 2>/dev/null || true
    rm -f "${PIDF}"
  fi
  pkill -f 'repro_circle_packing_26/keepalive_hot.py' >/dev/null 2>&1 || true
  pkill -f 'repro_circle_packing_26/keepalive.py' >/dev/null 2>&1 || true
  sleep 1

  export KEEPALIVE_URL="http://127.0.0.1:${FRONTEND_PORT}/v1/chat/completions"
  export KEEPALIVE_MODEL="${SERVED_NAME}"
  export KEEPALIVE_WORKERS="${KEEPALIVE_WORKERS:-8}"
  export KEEPALIVE_MAX_TOKENS=16
  export KEEPALIVE_SLEEP=1.0
  export no_proxy="*"
  export NO_PROXY="*"

  nohup "${PYTHON}" "${SIMPLES_DIR}/scripts/repro_circle_packing_26/keepalive.py" \
    </dev/null >> "${SIMPLES_DIR}/scripts/repro_circle_packing_26/logs/keepalive.log" 2>&1 &
  echo $! > "${PIDF}"
  disown || true
  echo "light_keepalive pid=$(cat "${PIDF}") workers=${KEEPALIVE_WORKERS} sleep=${KEEPALIVE_SLEEP}"
} > "${OUT}" 2>&1
