#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
ip="${KERNEL_NODE}"
log "launching gpukernel_zero server on ${ip}:${KERNEL_PORT}"
kernel_remote "${ip}" "bash ${HERE}/start_remote.sh ${KERNEL_LOG_DIR} ${PYTHON} ${SIMPLES_DIR} ${KERNEL_PORT}"
log "waiting for /health"
for _ in $(seq 1 60); do
  if curl --noproxy '*' -fsS "http://${ip}:${KERNEL_PORT}/health" >/dev/null 2>&1; then
    log "kernel_zero server healthy at ${ip}:${KERNEL_PORT}"
    exit 0
  fi
  sleep 5
done
log "server failed to become healthy; last log:"
tail -n 40 "${KERNEL_LOG_DIR}/kernel_server.log" || true
exit 1
