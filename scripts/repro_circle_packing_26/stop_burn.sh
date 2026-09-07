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

TARGETS=("${@:-${ALL_IPS[@]}}")
for ip in "${TARGETS[@]}"; do
  remote "${ip}" 'if [[ -f /tmp/gpu_burn.pid ]]; then kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true; rm -f /tmp/gpu_burn.pid; fi; pkill -f "python3? .*/burn_node.py" >/dev/null 2>&1 || true'
  log "burn stopped on ${ip}"
done
