#!/usr/bin/env bash
# Point nginx at the 24 remaining vLLM replicas (exclude KERNEL_NODE).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

export NODE_IP_LIST="${LLM_IPS}"
# Re-source so ALL_IPS is rebuilt from NODE_IP_LIST.
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"
export ALL_IPS
bash "${PACKING_DIR}/gen_nginx_conf.sh"
if [[ -f "${LOG_DIR}/nginx.pid" ]] && kill -0 "$(cat "${LOG_DIR}/nginx.pid")" 2>/dev/null; then
  nginx -s reload -c "${PACKING_DIR}/nginx.conf" -p "${PACKING_DIR}/"
  echo "nginx reloaded on :${FRONTEND_PORT} backends=${ALL_IPS[*]}"
else
  nginx -c "${PACKING_DIR}/nginx.conf" -p "${PACKING_DIR}/"
  echo "nginx started on :${FRONTEND_PORT} backends=${ALL_IPS[*]}"
fi
