#!/usr/bin/env bash
# Resume the unguarded paper-scale run from instance-bd60257b.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMPLES="/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES"
INSTANCE="${1:-checkpoints/gpukernel_zero_trimul/2026-09-08/instance-bd60257b}"
shift || true
if [[ "${INSTANCE}" != /* ]]; then
  INSTANCE="${SIMPLES}/${INSTANCE}"
fi
# SimpleTES auto-picks db_state_* by name; that prefers 165545 over later 043430.
if [[ -d "${INSTANCE}" && ! "$(basename "${INSTANCE}")" =~ ^db_state_ ]]; then
  CKPT="$(python3 "${HERE}/pick_latest_ckpt.py" "${INSTANCE}")"
  echo "[resume] using ${CKPT}" >&2
  INSTANCE="${CKPT}"
fi
exec bash "${HERE}/run_simpletes.sh" --resume "${INSTANCE}" --eval-concurrency 24 "$@"
