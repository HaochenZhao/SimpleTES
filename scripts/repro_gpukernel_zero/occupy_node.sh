#!/usr/bin/env bash
# Run ON a node. If GPUs look idle, start burn_node.py.
set -euo pipefail
LOG_DIR="${1:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero/logs}"
PYTHON="${2:-/opt/conda/envs/torch-base/bin/python}"
BURN="${3:-/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_circle_packing_26/burn_node.py}"
TAG="${4:-unknown}"
mkdir -p "${LOG_DIR}"
SMI="${LOG_DIR}/smi_${TAG}.txt"
{
  date '+%F %T'
  hostname
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  echo '---procs---'
  ps -eo pid,etime,cmd | grep -E 'vllm|launch_server|burn_node|keepalive|main.py' | grep -v grep || true
} > "${SMI}" 2>&1

max_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m}')"
echo "max_mem_mib=${max_mib}" >> "${SMI}"

if [[ -f /tmp/gpu_burn.pid ]] && kill -0 "$(cat /tmp/gpu_burn.pid)" 2>/dev/null; then
  echo "status=already_burning pid=$(cat /tmp/gpu_burn.pid)" >> "${SMI}"
  exit 0
fi

# Busy serving / eval already occupies cards; do not stack 40GiB burn on them.
if [[ "${max_mib}" -gt 8000 ]]; then
  echo "status=skip_busy" >> "${SMI}"
  exit 0
fi

if [[ -f /tmp/gpu_burn.pid ]]; then
  kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true
  rm -f /tmp/gpu_burn.pid
fi
export BURN_HOLD_GIB="${BURN_HOLD_GIB:-40}"
export BURN_MATRIX="${BURN_MATRIX:-8192}"
nohup "${PYTHON}" "${BURN}" </dev/null > "${LOG_DIR}/burn_${TAG}.log" 2>&1 &
echo $! > /tmp/gpu_burn.pid
disown || true
echo "status=started pid=$(cat /tmp/gpu_burn.pid)" >> "${SMI}"
