#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
OUT="${KERNEL_LOG_DIR}/kick_resume.out"
{
  echo "[$(date '+%F %T')] prep kernel node"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o ServerAliveInterval=3 -o ServerAliveCountMax=3 \
    "root@${KERNEL_NODE}" \
    "bash ${HERE}/prep_kernel_resume.sh ${KERNEL_LOG_DIR} ${PYTHON} ${SIMPLES_DIR} ${KERNEL_PORT}"

  echo "[$(date '+%F %T')] health"
  export no_proxy='*'
  for i in $(seq 1 30); do
    if curl --noproxy '*' -fsS --max-time 8 "http://${KERNEL_NODE}:${KERNEL_PORT}/health"; then
      echo
      echo "health_ok after ${i}"
      break
    fi
    echo "health_wait ${i}"
    sleep 2
  done

  echo "[$(date '+%F %T')] start resume"
  cd "${SIMPLES_DIR}"
  nohup bash "${HERE}/resume_simpletes.sh" \
    > "${KERNEL_LOG_DIR}/resume.log" 2>&1 &
  echo $! > "${KERNEL_LOG_DIR}/resume.pid"
  echo "resume_pid=$(cat "${KERNEL_LOG_DIR}/resume.pid")"
  echo KICK_RESUME_DONE
} > "${OUT}" 2>&1
