#!/usr/bin/env bash
# Run ON a kernel node. Start official (guarded) gpukernel launch_server with a GPU split.
set -euo pipefail
LOG_DIR="${1:?log dir}"
PYTHON="${2:?python}"
SIMPLES_DIR="${3:?simpletes dir}"
PORT="${4:-8000}"
COMPILE_GPUS="${5:-0,1,2,3}"
EVAL_GPUS="${6:-4,5,6,7}"
TAG="${7:-node}"
mkdir -p "${LOG_DIR}"

if [[ -f /tmp/gpu_burn.pid ]]; then
  kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true
  rm -f /tmp/gpu_burn.pid
fi
pkill -f 'repro_circle_packing_26/burn_node.py' >/dev/null 2>&1 || true

if [[ -f /tmp/kernel_server.pid ]]; then
  kill "$(cat /tmp/kernel_server.pid)" 2>/dev/null || true
  rm -f /tmp/kernel_server.pid
fi
while read -r pid; do
  [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
done < <(pgrep -f 'datasets/gpukernel_zero/server/launch_server.py' || true)
while read -r pid; do
  [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
done < <(pgrep -f 'datasets/gpukernel/server/launch_server.py' || true)
sleep 1

for _ in $(seq 1 90); do
  max_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print int(m)}')"
  if [[ "${max_mib}" -lt 4000 ]]; then
    echo "gpus_free max_mib=${max_mib}"
    break
  fi
  echo "waiting_gpu_free max_mib=${max_mib}"
  sleep 2
done

LOG="/tmp/kernel_guarded_${TAG}.log"
if [[ -f "${LOG}" ]]; then
  mv "${LOG}" "${LOG}.prev.$(date +%Y%m%d_%H%M%S)"
fi
ln -sfn "${LOG}" "${LOG_DIR}/kernel_${TAG}.log" 2>/dev/null || true

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Inherit from caller; paper bringup sets this so the naive PyTorch seed can score.
export SIMPLES_KERNEL_ALLOW_TORCH_IMPL="${SIMPLES_KERNEL_ALLOW_TORCH_IMPL:-}"
cd "${SIMPLES_DIR}"
nohup "${PYTHON}" datasets/gpukernel/server/launch_server.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --backend triton \
  --triton-compiler-gpus "${COMPILE_GPUS}" \
  --triton-eval-gpus "${EVAL_GPUS}" \
  --triton-compiler-gpu-concurrency 4 \
  --timeout 1800 \
  </dev/null > "${LOG}" 2>&1 &
echo $! > /tmp/kernel_server.pid
disown || true
echo "pid=$(cat /tmp/kernel_server.pid) compile=${COMPILE_GPUS} eval=${EVAL_GPUS} server=gpukernel"
