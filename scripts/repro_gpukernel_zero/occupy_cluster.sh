#!/usr/bin/env bash
# Occupy idle GPUs on all four nodes. Busy vLLM/kernel nodes are left alone.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
OCCUPY="${HERE}/occupy_node.sh"
BURN="${SIMPLES_DIR}/scripts/repro_circle_packing_26/burn_node.py"
IPS=(28.33.242.47 28.33.244.15 28.33.244.19 28.33.243.17)

for ip in "${IPS[@]}"; do
  echo "[$(date '+%H:%M:%S')] occupy ${ip}"
  if [[ "${ip}" == "${HEAD_IP}" ]]; then
    bash "${OCCUPY}" "${KERNEL_LOG_DIR}" "${PYTHON}" "${BURN}" "${ip}" || true
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
      -o ServerAliveInterval=3 -o ServerAliveCountMax=2 \
      "root@${ip}" \
      "bash ${OCCUPY} ${KERNEL_LOG_DIR} ${PYTHON} ${BURN} ${ip}" || echo "ssh_fail ${ip}"
  fi
done

# If local nginx+vLLM still answers, put keepalive back.
if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:${FRONTEND_PORT}/v1/models" >/dev/null 2>&1; then
  if ! pgrep -f 'keepalive.py' >/dev/null 2>&1; then
    nohup "${PYTHON}" "${SIMPLES_DIR}/scripts/repro_circle_packing_26/keepalive.py" \
      </dev/null >> "${SIMPLES_DIR}/scripts/repro_circle_packing_26/logs/keepalive.log" 2>&1 &
    echo $! > "${KERNEL_LOG_DIR}/keepalive.pid"
    echo "keepalive restarted pid=$!"
  else
    echo "keepalive already running"
  fi
else
  echo "vLLM frontend down; burn-only occupancy"
fi
echo DONE
