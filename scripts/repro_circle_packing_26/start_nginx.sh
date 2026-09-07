#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
bash "${HERE}/gen_nginx_conf.sh"
nginx -s quit >/dev/null 2>&1 || true
# Do not touch the system default nginx; use our config only.
if [[ -f "${LOG_DIR}/nginx.pid" ]]; then
  kill "$(cat "${LOG_DIR}/nginx.pid")" 2>/dev/null || true
fi
nginx -c "${HERE}/nginx.conf" -p "${HERE}/"
echo "nginx listening on :${FRONTEND_PORT}"
