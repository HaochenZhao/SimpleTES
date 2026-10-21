#!/usr/bin/env bash
# nginx :8000 -> only launcher 8 vLLM replicas.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
export NODE_IP_LIST="${LLM_IPS:-${HEAD_IP}}"
# shellcheck source=../repro_circle_packing_26/cluster_env.sh
source "${PACKING_DIR}/cluster_env.sh"
export ALL_IPS
bash "${PACKING_DIR}/gen_nginx_conf.sh"
if [[ -f "${LOG_DIR}/nginx.pid" ]] && kill -0 "$(cat "${LOG_DIR}/nginx.pid")" 2>/dev/null; then
  nginx -s reload -c "${PACKING_DIR}/nginx.conf" -p "${PACKING_DIR}/"
else
  nginx -c "${PACKING_DIR}/nginx.conf" -p "${PACKING_DIR}/"
fi
echo "nginx launcher-only backends=${ALL_IPS[*]}"
