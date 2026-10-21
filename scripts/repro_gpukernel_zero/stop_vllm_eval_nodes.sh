#!/usr/bin/env bash
# Stop vLLM on the two nodes that will become kernel servers; hold GPUs with burn.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
BURN="${PACKING_DIR}/burn_node.py"
EVAL_LLM_IPS=(28.33.244.15 28.33.244.19)

ssh_cmd() {
  local ip="$1"
  shift
  # shellcheck disable=SC2086
  timeout 240 ssh ${SSH_OPTS} -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
    "root@${ip}" "$@"
}

for ip in "${EVAL_LLM_IPS[@]}"; do
  (
    echo "[${ip}] stop vLLM + burn"
    ssh_cmd "${ip}" "bash ${HERE}/remote_stop_vllm.sh ${KERNEL_LOG_DIR} ${PYTHON} ${BURN} ${ip}" \
      > "${KERNEL_LOG_DIR}/stop_vllm_${ip}.out" 2>&1 \
      || echo "[${ip}] stop_ssh_failed" >> "${KERNEL_LOG_DIR}/stop_vllm_${ip}.out"
  ) &
done
wait
echo "vllm_eval_nodes_stopped"
