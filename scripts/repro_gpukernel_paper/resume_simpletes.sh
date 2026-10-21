#!/usr/bin/env bash
# Resume paper-aligned TriMul from the latest db_state of an instance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
INSTANCE="${1:-checkpoints/gpukernel_trimul_paper/2026-09-17/instance-7bf43cb3}"
shift || true
if [[ "${INSTANCE}" != /* ]]; then
  INSTANCE="${SIMPLES_DIR}/${INSTANCE}"
fi
if [[ -d "${INSTANCE}" && ! "$(basename "${INSTANCE}")" =~ ^db_state_ ]]; then
  CKPT="$(python3 "${HERE}/pick_latest_ckpt.py" "${INSTANCE}")"
  echo "[resume] using ${CKPT}" >&2
  INSTANCE="${CKPT}"
fi
# Round-2 best-restart lives in a different instance and init program.
# Defaulting to run_simpletes.sh would swap the seed back to naive.
RUNNER="${HERE}/run_simpletes.sh"
if [[ "${INSTANCE}" == *instance-d3ff8a91* || "${INSTANCE}" == *2026-09-21* ]]; then
  RUNNER="${HERE}/run_simpletes_round2.sh"
fi
exec bash "${RUNNER}" --resume "${INSTANCE}" "$@"
