#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() {
  local ip="$1"
  shift
  if [[ "${ip}" == "${HEAD_IP}" ]]; then
    bash -lc "$*"
  else
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "root@${ip}" "$*"
  fi
}

stop_on() {
  local ip="$1"
  remote "${ip}" 'if [[ -f /tmp/gpu_burn.pid ]]; then kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true; rm -f /tmp/gpu_burn.pid; fi; pkill -f "python3? .*/burn_node.py" >/dev/null 2>&1 || true'
}

start_on() {
  local ip="$1"
  stop_on "${ip}"
  remote "${ip}" "mkdir -p ${LOG_DIR}; BURN_HOLD_GIB=40 BURN_MATRIX=8192 nohup ${PYTHON} ${HERE}/burn_node.py > ${LOG_DIR}/burn_${ip}.log 2>&1 & echo \$! > /tmp/gpu_burn.pid"
  log "burn started on ${ip}"
}

for ip in "${ALL_IPS[@]}"; do
  start_on "${ip}"
done
log "gpu burn running on ${#ALL_IPS[@]} nodes"
