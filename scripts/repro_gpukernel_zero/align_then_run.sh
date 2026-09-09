#!/usr/bin/env bash
# Align official best on the zero server, record the naive seed, then start search.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
cd "${SIMPLES_DIR}"

export no_proxy="*"
export NO_PROXY="*"
mkdir -p "${HERE}/results" "${KERNEL_LOG_DIR}"

PY="${PYTHON:-/opt/conda/envs/torch-base/bin/python}"
SMOKE="${HERE}/smoke_eval.py"

echo "[$(date '+%F %T')] smoke official trimul_best.py"
"${PY}" "${SMOKE}" \
  --program "${SIMPLES_DIR}/best_results/ai_foundations/trimul/trimul_best.py" \
  --out "${HERE}/results/h20_official_best.json"

echo "[$(date '+%F %T')] smoke gpukernel_zero seed"
"${PY}" "${SMOKE}" \
  --program "${SIMPLES_DIR}/datasets/gpukernel_zero/trimul/init_program.py" \
  --out "${HERE}/results/h20_seed.json"

echo "[$(date '+%F %T')] start unguarded paper-scale SimpleTES"
exec bash "${HERE}/run_simpletes.sh"
