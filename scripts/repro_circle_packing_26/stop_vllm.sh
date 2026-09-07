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

for ip in "${ALL_IPS[@]}"; do
  remote "${ip}" 'for g in $(seq 0 7); do
    if [[ -f /tmp/vllm_gpu_${g}.pid ]]; then
      kill "$(cat /tmp/vllm_gpu_${g}.pid)" 2>/dev/null || true
      rm -f /tmp/vllm_gpu_${g}.pid
    fi
  done
  pkill -f "vllm serve" >/dev/null 2>&1 || true'
  log "vLLM stopped on ${ip}"
done
