#!/usr/bin/env bash
# Hold all 4 nodes' GPUs until vLLM/kernel take over.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
OCCUPY="${HERE}/occupy_node.sh"
BURN="${SIMPLES_DIR}/scripts/repro_circle_packing_26/burn_node.py"
OUT="${KERNEL_LOG_DIR}/occupy_now.out"
{
  echo "[$(date '+%F %T')] occupy ALL_IPS=${ALL_IPS[*]} HEAD=${HEAD_IP}"
  for ip in "${ALL_IPS[@]}"; do
    (
      if [[ "${ip}" == "${HEAD_IP}" ]]; then
        bash "${OCCUPY}" "${KERNEL_LOG_DIR}" "${PYTHON}" "${BURN}" "${ip}"
      else
        timeout 40 ssh ${SSH_OPTS} -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
          "root@${ip}" "bash ${OCCUPY} ${KERNEL_LOG_DIR} ${PYTHON} ${BURN} ${ip}"
      fi
      echo "[$(date '+%H:%M:%S')] occupy_done ${ip} rc=$?"
    ) &
  done
  wait
  echo OCCUPY_NOW_DONE
} > "${OUT}" 2>&1
