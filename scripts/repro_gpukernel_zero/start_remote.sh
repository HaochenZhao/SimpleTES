#!/usr/bin/env bash
# Runs on KERNEL_NODE. Keep launch_server.py out of the SSH command line
# so pkill -f does not kill the starter.
set -euo pipefail
LOG_DIR="${1:?log dir}"
PYTHON="${2:?python}"
SIMPLES_DIR="${3:?simpletes dir}"
PORT="${4:-8000}"
mkdir -p "${LOG_DIR}"
if [[ -f /tmp/kernel_server.pid ]]; then
  kill "$(cat /tmp/kernel_server.pid)" 2>/dev/null || true
  rm -f /tmp/kernel_server.pid
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "${SIMPLES_DIR}"
nohup "${PYTHON}" datasets/gpukernel_zero/server/launch_server.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --backend triton \
  --triton-compiler-gpus 0,1 \
  --triton-eval-gpus 2,3,4,5,6,7 \
  --triton-compiler-gpu-concurrency 1 \
  --timeout 1800 \
  </dev/null > "${LOG_DIR}/kernel_server.log" 2>&1 &
echo $! > /tmp/kernel_server.pid
disown || true
echo "pid=$(cat /tmp/kernel_server.pid)"
