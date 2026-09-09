#!/usr/bin/env bash
# Wait for KERNEL_NODE GPU memory to drop, stop burn, start the Triton eval server.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ip="${KERNEL_NODE}"
log "waiting for vLLM memory to release on ${ip}"
for _ in $(seq 1 60); do
  used=$(kernel_remote "${ip}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=\$1} END {print int(s/8)}'")
  log "avg memory.used=${used} MiB"
  if [[ "${used}" -lt 4000 ]]; then
    break
  fi
  sleep 10
done

log "stopping burn on ${ip}"
bash "${PACKING_DIR}/stop_burn.sh" "${ip}"

log "launching kernel eval server on ${ip}:${KERNEL_PORT}"
kernel_remote "${ip}" "mkdir -p ${KERNEL_LOG_DIR}
if [[ -f /tmp/kernel_server.pid ]]; then
  kill \"\$(cat /tmp/kernel_server.pid)\" 2>/dev/null || true
  rm -f /tmp/kernel_server.pid
fi
export SIMPLES_KERNEL_ALLOW_TORCH_IMPL=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ${SIMPLES_DIR}
nohup ${PYTHON} datasets/gpukernel/server/launch_server.py \
  --host 0.0.0.0 \
  --port ${KERNEL_PORT} \
  --backend triton \
  --triton-compiler-gpus 0,1 \
  --triton-eval-gpus 2,3,4,5,6,7 \
  --triton-compiler-gpu-concurrency 1 \
  --timeout 1800 \
  > ${KERNEL_LOG_DIR}/kernel_server_${ip}.log 2>&1 &
echo \$! > /tmp/kernel_server.pid
"

log "waiting for /health"
for _ in $(seq 1 60); do
  if curl --noproxy '*' -fsS "http://${ip}:${KERNEL_PORT}/health" >/dev/null 2>&1; then
    log "kernel server healthy at ${ip}:${KERNEL_PORT}"
    exit 0
  fi
  sleep 5
done
log "kernel server failed to become healthy; last log:"
tail -n 40 "${KERNEL_LOG_DIR}/kernel_server_${ip}.log" || true
exit 1
