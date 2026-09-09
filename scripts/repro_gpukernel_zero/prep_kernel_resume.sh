#!/usr/bin/env bash
# Run ON 28.33.243.17: drop burn, recycle the wedged eval server.
set -euo pipefail
LOG_DIR="${1:?log dir}"
PYTHON="${2:?python}"
SIMPLES_DIR="${3:?simpletes dir}"
PORT="${4:-8000}"
mkdir -p "${LOG_DIR}"

if [[ -f /tmp/gpu_burn.pid ]]; then
  kill "$(cat /tmp/gpu_burn.pid)" 2>/dev/null || true
  rm -f /tmp/gpu_burn.pid
fi
# Match the burn script path only; do not put this string on an SSH argv.
pkill -f 'repro_circle_packing_26/burn_node.py' >/dev/null 2>&1 || true

if [[ -f /tmp/kernel_server.pid ]]; then
  kill "$(cat /tmp/kernel_server.pid)" 2>/dev/null || true
  rm -f /tmp/kernel_server.pid
fi
# Recycle leftover server by pid, not by matching this script's command line.
while read -r pid; do
  [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
done < <(pgrep -f 'datasets/gpukernel_zero/server/launch_server.py' || true)

for _ in $(seq 1 20); do
  max_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print int(m)}')"
  if [[ "${max_mib}" -lt 3000 ]]; then
    echo "gpus_free max_mib=${max_mib}"
    break
  fi
  echo "waiting_gpu_free max_mib=${max_mib}"
  sleep 2
done

if [[ -f "${LOG_DIR}/kernel_server.log" ]]; then
  mv "${LOG_DIR}/kernel_server.log" "${LOG_DIR}/kernel_server.log.prev.$(date +%Y%m%d_%H%M%S)"
fi

bash "${SIMPLES_DIR}/scripts/repro_gpukernel_zero/start_remote.sh" \
  "${LOG_DIR}" "${PYTHON}" "${SIMPLES_DIR}" "${PORT}"
echo PREP_DONE
