#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

CONF="${HERE}/nginx.conf"
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
  for ip in "${ALL_IPS[@]}"; do
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
      echo "    server ${ip}:$((BASE_PORT + gpu)) max_fails=20 fail_timeout=5s;"
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
echo "wrote ${CONF}"
