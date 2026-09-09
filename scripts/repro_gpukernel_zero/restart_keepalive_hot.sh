#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
KA="${SIMPLES_DIR}/scripts/repro_circle_packing_26/keepalive_hot.py"
LOG="${SIMPLES_DIR}/scripts/repro_circle_packing_26/logs/keepalive_hot.log"
PIDF="${KERNEL_LOG_DIR}/keepalive.pid"
mkdir -p "${KERNEL_LOG_DIR}"

# Stop the light keepalive and any previous hot loop.
if [[ -f "${PIDF}" ]]; then
  kill "$(cat "${PIDF}")" 2>/dev/null || true
  rm -f "${PIDF}"
fi
pkill -f 'repro_circle_packing_26/keepalive.py' >/dev/null 2>&1 || true
pkill -f 'repro_circle_packing_26/keepalive_hot.py' >/dev/null 2>&1 || true
sleep 1

export KEEPALIVE_MODEL="${SERVED_NAME}"
export KEEPALIVE_PER_REPLICA="${KEEPALIVE_PER_REPLICA:-6}"
export KEEPALIVE_MAX_TOKENS="${KEEPALIVE_MAX_TOKENS:-512}"
export no_proxy="*"
export NO_PROXY="*"

nohup "${PYTHON}" "${KA}" </dev/null >> "${LOG}" 2>&1 &
echo $! > "${PIDF}"
disown || true
echo "keepalive_hot started pid=$(cat "${PIDF}") per_replica=${KEEPALIVE_PER_REPLICA} max_tokens=${KEEPALIVE_MAX_TOKENS}"
