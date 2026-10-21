#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
OUT="${SIMPLES_DIR}/datasets/gpukernel_zero/server_info.yaml"
{
  echo "# gpukernel_zero: 3 exclusive eval nodes. Launcher ${HEAD_IP} stays on vLLM."
  echo "servers:"
  IFS=',' read -r -a nodes <<< "${KERNEL_NODES}"
  for ip in "${nodes[@]}"; do
    echo "  - ip: ${ip}"
    echo "    port: ${KERNEL_PORT}"
  done
  echo
  echo "load_balance: random"
  echo "request_timeout: 3000"
} > "${OUT}"
echo "wrote ${OUT} kernels=${KERNEL_NODES}"
