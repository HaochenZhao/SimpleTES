#!/usr/bin/env bash
# Stop vLLM only on KERNEL_NODE, then immediately burn so the node is not idle.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ip="${KERNEL_NODE}"
log "stopping vLLM on dedicated kernel node ${ip}"
kernel_remote "${ip}" 'for g in $(seq 0 7); do
  if [[ -f /tmp/vllm_gpu_${g}.pid ]]; then
    kill "$(cat /tmp/vllm_gpu_${g}.pid)" 2>/dev/null || true
    rm -f /tmp/vllm_gpu_${g}.pid
  fi
done
pkill -f "vllm serve" >/dev/null 2>&1 || true'

log "starting burn on ${ip} until kernel server is up"
kernel_remote "${ip}" "mkdir -p ${KERNEL_LOG_DIR}; BURN_HOLD_GIB=40 BURN_MATRIX=8192 nohup ${PYTHON} ${PACKING_DIR}/burn_node.py > ${KERNEL_LOG_DIR}/burn_${ip}.log 2>&1 & echo \$! > /tmp/gpu_burn.pid"
log "vLLM stopped and burn running on ${ip}"
