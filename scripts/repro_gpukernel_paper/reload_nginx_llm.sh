#!/usr/bin/env bash
# nginx :8000 -> TP-sharded vLLM replicas on LLM_IPS (not 8 ports/node).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
NREP="${VLLM_REPLICAS_PER_NODE:-4}"
CONF="${PACKING_DIR}/nginx.conf"
{
  cat <<EOF
pid ${LOG_DIR}/nginx.pid;
error_log ${LOG_DIR}/nginx_error.log warn;
worker_processes 4;
events { worker_connections 4096; }
http {
  access_log ${LOG_DIR}/nginx_access.log;
  upstream vllm_backends {
    least_conn;
EOF
  IFS=',' read -r -a llm <<< "${LLM_IPS}"
  for ip in "${llm[@]}"; do
    for r in $(seq 0 $((NREP - 1))); do
      echo "    server ${ip}:$((BASE_PORT + r)) max_fails=20 fail_timeout=5s;"
    done
  done
  cat <<EOF
  }
  server {
    listen ${FRONTEND_PORT};
    server_name _;
    location / {
      proxy_pass http://vllm_backends;
      proxy_http_version 1.1;
      proxy_set_header Host \$host;
      proxy_set_header Connection "";
      proxy_read_timeout 3600s;
      proxy_send_timeout 3600s;
      proxy_buffering off;
    }
  }
}
EOF
} > "${CONF}"
echo "wrote ${CONF} replicas_per_node=${NREP} llm=${LLM_IPS}"

if [[ -f "${LOG_DIR}/nginx.pid" ]] && kill -0 "$(cat "${LOG_DIR}/nginx.pid")" 2>/dev/null; then
  nginx -s reload -c "${CONF}" -p "${PACKING_DIR}/"
  echo "nginx reloaded"
else
  nginx -s quit >/dev/null 2>&1 || true
  nginx -c "${CONF}" -p "${PACKING_DIR}/"
  echo "nginx started"
fi
