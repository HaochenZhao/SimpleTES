#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
KA="${SIMPLES_DIR}/scripts/repro_circle_packing_26/keepalive.py"
LOG="${SIMPLES_DIR}/scripts/repro_circle_packing_26/logs/keepalive.log"
PIDF="${KERNEL_LOG_DIR}/keepalive.pid"
mkdir -p "${KERNEL_LOG_DIR}" "$(dirname "${LOG}")"

if [[ -f "${PIDF}" ]] && kill -0 "$(cat "${PIDF}")" 2>/dev/null; then
  echo "keepalive already pid=$(cat "${PIDF}")"
  exit 0
fi
if pgrep -f 'repro_circle_packing_26/keepalive.py' >/dev/null 2>&1; then
  echo "keepalive already running"
  exit 0
fi

export KEEPALIVE_URL="http://127.0.0.1:${FRONTEND_PORT}/v1/chat/completions"
export KEEPALIVE_MODEL="${SERVED_NAME}"
export KEEPALIVE_WORKERS="${KEEPALIVE_WORKERS:-32}"
export KEEPALIVE_MAX_TOKENS=16
export KEEPALIVE_SLEEP=0.4
export no_proxy="*"
export NO_PROXY="*"

nohup "${PYTHON}" "${KA}" </dev/null >> "${LOG}" 2>&1 &
echo $! > "${PIDF}"
disown || true
echo "keepalive started pid=$(cat "${PIDF}") workers=${KEEPALIVE_WORKERS}"
